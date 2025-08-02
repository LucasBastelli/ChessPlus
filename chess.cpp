#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <algorithm>
#include <sstream>
#include <limits>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#include <cstdint>

// ===================================================================
//
// 				  GEMINI CHESS BOT - VERSÃO APRIMORADA
//
// ===================================================================
// Principais melhorias:
// - Detecção de Empate (Repetição, 50 lances)
// - Busca com Aprofundamento Iterativo (Iterative Deepening)
// - Algoritmo de Busca PVS (Principal Variation Search)
// - Ordenação de Lances Avançada (MVV-LVA, Killers, History)
// - Null Move Pruning (NMP)
// - Estrutura de Avaliação com Fases de Jogo
// ===================================================================


// === Constantes e Estruturas Globais ===

using Bitboard = uint64_t;

// --- Tipos e Enumerações
enum PieceType { WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, NO_PIECE };
const int NUM_PIECE_TYPES = 12;
const int NUM_BASE_PIECE_TYPES = 6;
enum Color { WHITE, BLACK, BOTH };
enum Square {
    A1, B1, C1, D1, E1, F1, G1, H1, A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3, A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5, A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7, A8, B8, C8, D8, E8, F8, G8, H8, NO_SQ
};

// --- Estrutura do Lance
struct Move {
    Square fromSquare; Square toSquare; PieceType pieceMoved; PieceType pieceCaptured; PieceType promotedTo;
    bool isCapture; bool isEnPassant; bool isPromotion; bool isCastleKingside; bool isCastleQueenside;

    Move() : fromSquare(NO_SQ), pieceMoved(NO_PIECE), isCapture(false) {}
    Move(Square from, Square to, PieceType moved, PieceType captured = NO_PIECE, PieceType promoted = NO_PIECE,
         bool ep = false, bool castleK = false, bool castleQ = false)
        : fromSquare(from), toSquare(to), pieceMoved(moved), pieceCaptured(captured),
          promotedTo(promoted), isCapture(captured != NO_PIECE || ep), isEnPassant(ep),
          isPromotion(promoted != NO_PIECE), isCastleKingside(castleK), isCastleQueenside(castleQ) {}

    std::string toString() const;
    bool operator==(const Move& other) const;
};

// --- Estrutura da Posição
struct BoardState {
    std::array<Bitboard, NUM_PIECE_TYPES> pieceBitboards;
    std::array<Bitboard, 2> colorBitboards;
    Bitboard occupiedBitboard;
    Color sideToMove;
    int castlingRights;
    Square enPassantSquare;
    int halfMoveClock;
    int fullMoveNumber;
    Bitboard zobristKey;
};

// --- Constantes de Jogo e Busca
const int WK_CASTLE = 1, WQ_CASTLE = 2, BK_CASTLE = 4, BQ_CASTLE = 8;
const int PAWN_VALUE = 100, KNIGHT_VALUE = 320, BISHOP_VALUE = 330, ROOK_VALUE = 500, QUEEN_VALUE = 900, KING_VALUE_MATERIAL = 0;
const int MATE_SCORE = 20000, INFINITE_SCORE = 32000, DRAW_SCORE = 0;
const int MAX_PLY = 64; // Profundidade máxima de busca

// --- Bitboards de Arquivos e Fileiras
const Bitboard FILE_A_BB = 0x0101010101010101ULL;
const Bitboard FILE_B_BB = FILE_A_BB << 1;
const Bitboard FILE_C_BB = FILE_A_BB << 2;
const Bitboard FILE_D_BB = FILE_A_BB << 3;
const Bitboard FILE_E_BB = FILE_A_BB << 4;
const Bitboard FILE_F_BB = FILE_A_BB << 5;
const Bitboard FILE_G_BB = FILE_A_BB << 6;
const Bitboard FILE_H_BB = FILE_A_BB << 7;

const Bitboard RANK_1_BB = 0xFFULL;
const Bitboard RANK_2_BB = RANK_1_BB << (8 * 1);
const Bitboard RANK_3_BB = RANK_1_BB << (8 * 2);
const Bitboard RANK_4_BB = RANK_1_BB << (8 * 3);
const Bitboard RANK_5_BB = RANK_1_BB << (8 * 4);
const Bitboard RANK_6_BB = RANK_1_BB << (8 * 5);
const Bitboard RANK_7_BB = RANK_1_BB << (8 * 6);
const Bitboard RANK_8_BB = RANK_1_BB << (8 * 7);

const Bitboard NOT_A_FILE = ~FILE_A_BB;
const Bitboard NOT_H_FILE = ~FILE_H_BB;
const Bitboard NOT_HG_FILE = ~(FILE_H_BB | FILE_G_BB);
const Bitboard NOT_AB_FILE = ~(FILE_A_BB | FILE_B_BB);

// --- Tabelas Globais
std::array<int, NUM_BASE_PIECE_TYPES> pieceMaterialValue = { PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE_MATERIAL };
std::array<std::array<int, 64>, NUM_BASE_PIECE_TYPES> pieceSquareTables_Opening;
std::array<std::array<int, 64>, NUM_BASE_PIECE_TYPES> pieceSquareTables_Endgame;
std::array<Bitboard, 64> knightAttacks;
std::array<Bitboard, 64> kingAttacks;

// --- Zobrist Hashing
std::array<std::array<Bitboard, 64>, NUM_PIECE_TYPES> zobristPieceKeys;
Bitboard zobristSideToMoveKey;
std::array<Bitboard, 16> zobristCastlingKeys;
std::array<Bitboard, 64> zobristEnPassantKeys;

// --- Tabela de Transposição (TT) ---
enum TT_FLAG { TT_EXACT, TT_LOWER_BOUND, TT_UPPER_BOUND };
struct TTEntry {
    Bitboard key;
    int depth;
    int score;
    TT_FLAG flag;
    Move bestMove; // NOVO: Armazenar o melhor lance para ordenação
};
const int TT_SIZE = 1048576 * 4; // Aumentado para melhor performance
std::vector<TTEntry> transpositionTable(TT_SIZE);


// --- NOVO: Estruturas de Ordenação de Lances ---
struct ScoredMove {
    Move move;
    int score;
    ScoredMove(Move m, int s) : move(m), score(s) {}
    bool operator<(const ScoredMove& other) const { return score > other.score; }
};

// Heurísticas de ordenação
std::array<std::array<Move, 2>, MAX_PLY> killerMoves;
std::array<std::array<int, 64>, NUM_PIECE_TYPES> historyScores;

// --- NOVO: Histórico para Regra de Repetição ---
std::vector<Bitboard> gameHistory;

// --- Funções Utilitárias de Bitboard ---
void set_bit(Bitboard &bb, Square sq) { bb |= (1ULL << sq); }
void clear_bit(Bitboard &bb, Square sq) { bb &= ~(1ULL << sq); }
bool get_bit(Bitboard bb, Square sq) { return (bb & (1ULL << sq)) != 0; }

#if defined(__GNUC__) || defined(__clang__)
inline int get_lsb_index(Bitboard bb) { if (bb == 0) return -1; return __builtin_ctzll(bb); }
inline Square pop_lsb(Bitboard &bb) { if (bb == 0) return NO_SQ; Square lsb_sq = static_cast<Square>(__builtin_ctzll(bb)); bb &= (bb - 1); return lsb_sq; }
inline int countSetBits(Bitboard bb) { return __builtin_popcountll(bb); }
#elif defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanForward64)
#pragma intrinsic(__popcnt64)
inline int get_lsb_index(Bitboard bb) { if (bb == 0) return -1; unsigned long index; _BitScanForward64(&index, bb); return static_cast<int>(index); }
inline Square pop_lsb(Bitboard &bb) { if (bb == 0) return NO_SQ; unsigned long index; _BitScanForward64(&index, bb); bb &= (bb - 1); return static_cast<Square>(index); }
inline int countSetBits(Bitboard bb) { return static_cast<int>(__popcnt64(bb)); }
#else
#warning "Compilador não-otimizado. Performance será severamente impactada."
// Fallbacks lentos
#endif

inline Square mirror_square(Square sq) { return static_cast<Square>(sq ^ 56); }

// --- Protótipos de Funções ---
void initialize_all();
void initialize_zobrist_keys();
void initialize_attack_tables();
void initialize_evaluation_parameters();
void initialize_board(BoardState &board);
Bitboard generate_zobrist_key(const BoardState& board);
void clear_transposition_table();
void clear_search_heuristics();

void make_move(BoardState &board, const Move &move, bool inSearch);
bool is_square_attacked(Square sq, Color attackerColor, const BoardState &board);
bool is_king_in_check(const BoardState &board, Color kingColor);
bool is_draw_by_repetition(const BoardState& board);

void generate_legal_moves(const BoardState &original_board, std::vector<Move> &legalMoveList, bool capturesOnly);
void generate_pawn_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_knight_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_bishop_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_rook_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_queen_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_king_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);

int evaluate(const BoardState &board);
int quiescence_search(BoardState currentBoard, int alpha, int beta, int ply);
int search(BoardState currentBoard, int depth, int alpha, int beta, int ply, bool isPV, bool isNull);
Move find_best_move(BoardState& board, int maxDepth, int timeLimitMs);

void print_pretty_board(const BoardState &board);
char piece_to_char(PieceType pt);
Move parse_move_input(const std::string& moveStr, const std::vector<Move>& legalMoves, Color currentSide);

// --- Implementações ---

std::string Move::toString() const {
    if (fromSquare == NO_SQ) return "invalid_move";
    std::string moveStr;
    moveStr += static_cast<char>('a' + (fromSquare % 8));
    moveStr += static_cast<char>('1' + (fromSquare / 8));
    moveStr += static_cast<char>('a' + (toSquare % 8));
    moveStr += static_cast<char>('1' + (toSquare / 8));
    if (isPromotion) {
        char promoChar = ' ';
        PieceType basePromoPiece = static_cast<PieceType>(promotedTo % NUM_BASE_PIECE_TYPES);
        switch(basePromoPiece) {
            case WQ: promoChar = 'q'; break; case WR: promoChar = 'r'; break;
            case WB: promoChar = 'b'; break; case WN: promoChar = 'n'; break;
            default: break;
        }
        moveStr += promoChar;
    }
    return moveStr;
}

bool Move::operator==(const Move& other) const {
    return fromSquare == other.fromSquare && toSquare == other.toSquare && promotedTo == other.promotedTo;
}


void initialize_all() {
    initialize_zobrist_keys();
    initialize_attack_tables();
    initialize_evaluation_parameters();
    clear_transposition_table();
}

void clear_transposition_table() {
    for(auto& entry : transpositionTable) {
        entry = {0, 0, 0, TT_EXACT, Move()};
    }
}

void clear_search_heuristics() {
    for (auto& ply_killers : killerMoves) {
        ply_killers[0] = Move();
        ply_killers[1] = Move();
    }
    for (auto& piece_history : historyScores) {
        piece_history.fill(0);
    }
}

void initialize_zobrist_keys() {
    std::mt19937_64 randomEngine(123456789);
    std::uniform_int_distribution<Bitboard> dist(0, std::numeric_limits<Bitboard>::max());
    for (int p = 0; p < NUM_PIECE_TYPES; ++p) for (int s = 0; s < 64; ++s) zobristPieceKeys[p][s] = dist(randomEngine);
    zobristSideToMoveKey = dist(randomEngine);
    for (int i = 0; i < 16; ++i) zobristCastlingKeys[i] = dist(randomEngine);
    for (int i = 0; i < 64; ++i) zobristEnPassantKeys[i] = dist(randomEngine);
}

Bitboard generate_zobrist_key(const BoardState& board) {
    Bitboard key = 0;
    for (int piece = 0; piece < NUM_PIECE_TYPES; ++piece) {
        Bitboard bb = board.pieceBitboards[piece];
        while (bb) {
            Square sq = pop_lsb(bb);
            key ^= zobristPieceKeys[piece][sq];
        }
    }
    if (board.sideToMove == BLACK) key ^= zobristSideToMoveKey;
    key ^= zobristCastlingKeys[board.castlingRights];
    if (board.enPassantSquare != NO_SQ) key ^= zobristEnPassantKeys[board.enPassantSquare];
    return key;
}

void initialize_evaluation_parameters() {
    // Mesmas tabelas de antes, mas separadas para abertura e final
    pieceSquareTables_Opening[WP] = {
        0,  0,  0,  0,  0,  0,  0,  0, 50, 50, 50, 50, 50, 50, 50, 50,
       10, 10, 20, 30, 30, 20, 10, 10,  5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,  5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,  0,  0,  0,  0,  0,  0,  0,  0
    };
    pieceSquareTables_Endgame[WP] = { // Peões são mais valiosos no final
        0,  0,  0,  0,  0,  0,  0,  0, 80, 80, 80, 80, 80, 80, 80, 80,
       50, 50, 50, 50, 50, 50, 50, 50, 30, 30, 30, 30, 30, 30, 30, 30,
       20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    };

    pieceSquareTables_Opening[WN] = {
       -50,-40,-30,-30,-30,-30,-40,-50, -40,-20,  0,  5,  5,  0,-20,-40,
       -30,  5, 10, 15, 15, 10,  5,-30, -30,  0, 15, 20, 20, 15,  0,-30,
       -30,  5, 15, 20, 20, 15,  5,-30, -30,  0, 10, 15, 15, 10,  0,-30,
       -40,-20,  0,  0,  0,  0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50
    };
    pieceSquareTables_Endgame[WN] = pieceSquareTables_Opening[WN];

    pieceSquareTables_Opening[WB] = {
       -20,-10,-10,-10,-10,-10,-10,-20, -10,  5,  0,  0,  0,  0,  5,-10,
       -10, 10, 10, 10, 10, 10, 10,-10, -10,  0, 10, 10, 10, 10,  0,-10,
       -10,  5,  5, 10, 10,  5,  5,-10, -10,  0,  5, 10, 10,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10, -20,-10,-10,-10,-10,-10,-10,-20
    };
    pieceSquareTables_Endgame[WB] = pieceSquareTables_Opening[WB];

    pieceSquareTables_Opening[WR] = {
        0,  0,  0,  5,  5,  0,  0,  0, -5,  0,  0,  0,  0,  0,  0, -5,
       -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5,
       -5,  0,  0,  0,  0,  0,  0, -5, -5,  0,  0,  0,  0,  0,  0, -5,
        5, 10, 10, 10, 10, 10, 10,  5,  0,  0,  0,  0,  0,  0,  0,  0
    };
    pieceSquareTables_Endgame[WR] = pieceSquareTables_Opening[WR];

    pieceSquareTables_Opening[WQ] = {
       -20,-10,-10, -5, -5,-10,-10,-20, -10,  0,  5,  0,  0,  0,  0,-10,
       -10,  5,  5,  5,  5,  5,  0,-10,   0,  0,  5,  5,  5,  5,  0, -5,
        -5,  0,  5,  5,  5,  5,  0, -5, -10,  0,  5,  5,  5,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10, -20,-10,-10, -5, -5,-10,-10,-20
    };
    pieceSquareTables_Endgame[WQ] = pieceSquareTables_Opening[WQ];

    pieceSquareTables_Opening[WK] = { // Rei seguro nos cantos na abertura
       -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
       -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,  20, 30, 10,  0,  0, 10, 30, 20
    };
    pieceSquareTables_Endgame[WK] = { // Rei ativo no centro no final
       -50,-30,-30,-30,-30,-30,-30,-50, -30,-30,  0,  0,  0,  0,-30,-30,
       -30,-10, 20, 30, 30, 20,-10,-30, -30,-10, 30, 40, 40, 30,-10,-30,
       -30,-10, 30, 40, 40, 30,-10,-30, -30,-10, 20, 30, 30, 20,-10,-30,
       -30,-30,  0,  0,  0,  0,-30,-30, -50,-30,-30,-30,-30,-30,-30,-50
    };
}

void initialize_attack_tables() {
    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        Bitboard spot = 1ULL << sq_idx;
        knightAttacks[sq_idx] = 0ULL;
        knightAttacks[sq_idx] |= (spot << 17) & NOT_A_FILE; knightAttacks[sq_idx] |= (spot << 15) & NOT_H_FILE;
        knightAttacks[sq_idx] |= (spot << 10) & NOT_AB_FILE; knightAttacks[sq_idx] |= (spot << 6) & NOT_HG_FILE;
        knightAttacks[sq_idx] |= (spot >> 17) & NOT_H_FILE; knightAttacks[sq_idx] |= (spot >> 15) & NOT_A_FILE;
        knightAttacks[sq_idx] |= (spot >> 10) & NOT_HG_FILE; knightAttacks[sq_idx] |= (spot >> 6) & NOT_AB_FILE;

        kingAttacks[sq_idx] = 0ULL;
        kingAttacks[sq_idx] |= (spot << 9) & NOT_A_FILE; kingAttacks[sq_idx] |= (spot << 8); kingAttacks[sq_idx] |= (spot << 7) & NOT_H_FILE;
        kingAttacks[sq_idx] |= (spot << 1) & NOT_A_FILE; kingAttacks[sq_idx] |= (spot >> 1) & NOT_H_FILE;
        kingAttacks[sq_idx] |= (spot >> 7) & NOT_A_FILE; kingAttacks[sq_idx] |= (spot >> 8); kingAttacks[sq_idx] |= (spot >> 9) & NOT_H_FILE;
    }
}
char piece_to_char(PieceType pt) {
    switch (pt) {
        case WP: return 'P'; case WN: return 'N'; case WB: return 'B'; case WR: return 'R'; case WQ: return 'Q'; case WK: return 'K';
        case BP: return 'p'; case BN: return 'n'; case BB: return 'b'; case BR: return 'r'; case BQ: return 'q'; case BK: return 'k';
        default: return ' ';
    }
}

// NOVO: make_move agora lida com o histórico do jogo
void make_move(BoardState &board, const Move &move, bool inSearch = false) {
    if (!inSearch) {
        gameHistory.push_back(board.zobristKey);
    }
    
    // Zobrist update (incremental)
    board.zobristKey ^= zobristCastlingKeys[board.castlingRights];
    if (board.enPassantSquare != NO_SQ) board.zobristKey ^= zobristEnPassantKeys[board.enPassantSquare];

    Color us = board.sideToMove;
    Color them = (us == WHITE) ? BLACK : WHITE;

    // --- Movendo a peça ---
    board.zobristKey ^= zobristPieceKeys[move.pieceMoved][move.fromSquare];
    clear_bit(board.pieceBitboards[move.pieceMoved], move.fromSquare);
    clear_bit(board.colorBitboards[us], move.fromSquare);

    if (move.isCapture) {
        Square capturedSq = move.toSquare;
        PieceType actualCapturedPiece = move.pieceCaptured;

        if (move.isEnPassant) {
            capturedSq = (us == WHITE) ? static_cast<Square>(move.toSquare - 8) : static_cast<Square>(move.toSquare + 8);
            actualCapturedPiece = (us == WHITE) ? BP : WP;
        }
        if (actualCapturedPiece != NO_PIECE) {
            board.zobristKey ^= zobristPieceKeys[actualCapturedPiece][capturedSq];
            clear_bit(board.pieceBitboards[actualCapturedPiece], capturedSq);
            clear_bit(board.colorBitboards[them], capturedSq);
        }
    }

    PieceType pieceToPlace = move.isPromotion ? move.promotedTo : move.pieceMoved;
    board.zobristKey ^= zobristPieceKeys[pieceToPlace][move.toSquare];
    set_bit(board.pieceBitboards[pieceToPlace], move.toSquare);
    set_bit(board.colorBitboards[us], move.toSquare);

    // --- Lógica de Roque ---
    if (move.isCastleKingside) {
        if (us == WHITE) {
            board.zobristKey ^= zobristPieceKeys[WR][H1]; board.zobristKey ^= zobristPieceKeys[WR][F1];
            clear_bit(board.pieceBitboards[WR], H1); clear_bit(board.colorBitboards[us], H1);
            set_bit(board.pieceBitboards[WR], F1);   set_bit(board.colorBitboards[us], F1);
        } else {
            board.zobristKey ^= zobristPieceKeys[BR][H8]; board.zobristKey ^= zobristPieceKeys[BR][F8];
            clear_bit(board.pieceBitboards[BR], H8); clear_bit(board.colorBitboards[us], H8);
            set_bit(board.pieceBitboards[BR], F8);   set_bit(board.colorBitboards[us], F8);
        }
    } else if (move.isCastleQueenside) {
        if (us == WHITE) {
            board.zobristKey ^= zobristPieceKeys[WR][A1]; board.zobristKey ^= zobristPieceKeys[WR][D1];
            clear_bit(board.pieceBitboards[WR], A1); clear_bit(board.colorBitboards[us], A1);
            set_bit(board.pieceBitboards[WR], D1);   set_bit(board.colorBitboards[us], D1);
        } else {
            board.zobristKey ^= zobristPieceKeys[BR][A8]; board.zobristKey ^= zobristPieceKeys[BR][D8];
            clear_bit(board.pieceBitboards[BR], A8); clear_bit(board.colorBitboards[us], A8);
            set_bit(board.pieceBitboards[BR], D8);   set_bit(board.colorBitboards[us], D8);
        }
    }

    // --- Atualizar direitos de roque ---
    if (move.pieceMoved == WK) board.castlingRights &= ~(WK_CASTLE | WQ_CASTLE);
    if (move.pieceMoved == BK) board.castlingRights &= ~(BK_CASTLE | BQ_CASTLE);
    if (move.fromSquare == H1 || move.toSquare == H1) board.castlingRights &= ~WK_CASTLE;
    if (move.fromSquare == A1 || move.toSquare == A1) board.castlingRights &= ~WQ_CASTLE;
    if (move.fromSquare == H8 || move.toSquare == H8) board.castlingRights &= ~BK_CASTLE;
    if (move.fromSquare == A8 || move.toSquare == A8) board.castlingRights &= ~BQ_CASTLE;

    // --- Atualizar En Passant ---
    board.enPassantSquare = NO_SQ;
    if (move.pieceMoved == WP && move.toSquare - move.fromSquare == 16) board.enPassantSquare = static_cast<Square>(move.fromSquare + 8);
    else if (move.pieceMoved == BP && move.fromSquare - move.toSquare == 16) board.enPassantSquare = static_cast<Square>(move.fromSquare - 8);

    // --- Atualizar Contadores e Chave Zobrist Final ---
    if (move.pieceMoved == WP || move.pieceMoved == BP || move.isCapture) {
        board.halfMoveClock = 0;
        if (!inSearch) gameHistory.clear(); // Lance irreversível, limpa histórico de repetição
    } else {
        board.halfMoveClock++;
    }
    if (us == BLACK) board.fullMoveNumber++;

    board.sideToMove = them;
    board.occupiedBitboard = board.colorBitboards[WHITE] | board.colorBitboards[BLACK];

    board.zobristKey ^= zobristCastlingKeys[board.castlingRights];
    if (board.enPassantSquare != NO_SQ) board.zobristKey ^= zobristEnPassantKeys[board.enPassantSquare];
    board.zobristKey ^= zobristSideToMoveKey;
}


bool is_king_in_check(const BoardState &board, Color kingColor) {
    PieceType kingPiece = (kingColor == WHITE) ? WK : BK;
    Bitboard kingBb = board.pieceBitboards[kingPiece];
    if (kingBb == 0) return true; // Rei capturado, posição ilegal
    Square kingSq = static_cast<Square>(get_lsb_index(kingBb));
    return is_square_attacked(kingSq, (kingColor == WHITE) ? BLACK : WHITE, board);
}


// --- Geração de Lances (Mesma lógica, agora funções auxiliares) ---
// (O código de geração de lances (pawn, knight, sliders, etc.) é idêntico ao original e foi omitido por brevidade, mas está incluído no bot final)
void generate_pawn_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) { /* ... implementação original ... */ }
void generate_knight_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) { /* ... implementação original ... */ }
void generate_sliding_pseudo_moves(const BoardState &board, std::vector<Move> &moveList, PieceType pt, const int directions[], int num_directions) { /* ... implementação original ... */ }
void generate_bishop_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) { /* ... implementação original ... */ }
void generate_rook_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) { /* ... implementação original ... */ }
void generate_queen_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) { /* ... implementação original ... */ }
void generate_king_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) { /* ... implementação original ... */ }
bool is_square_attacked(Square sq, Color attackerColor, const BoardState &board) { /* ... implementação original ... */ }


void generate_legal_moves(const BoardState &original_board, std::vector<Move> &legalMoveList, bool capturesOnly) {
    legalMoveList.clear();
    std::vector<Move> pseudoLegalMoves;
    // ... chamadas para as funções generate_*_pseudo_moves ...

    for (const auto& pseudoMove : pseudoLegalMoves) {
        if (capturesOnly && !pseudoMove.isCapture) continue;

        BoardState tempBoard = original_board;
        make_move(tempBoard, pseudoMove, true); // Usa o make_move na busca
        if (!is_king_in_check(tempBoard, original_board.sideToMove)) {
            legalMoveList.push_back(pseudoMove);
        }
    }
}

bool is_draw_by_repetition(const BoardState& board) {
    int count = 0;
    for (Bitboard old_key : gameHistory) {
        if (old_key == board.zobristKey) {
            count++;
        }
    }
    return count >= 2; // A posição atual + 2 no histórico = 3 ocorrências
}


int evaluate(const BoardState &board) {
    int materialScore = 0;
    int positionalScore = 0;
    int totalMaterial = 0;

    for (int pt_base_idx = 0; pt_base_idx < NUM_BASE_PIECE_TYPES; ++pt_base_idx) {
        totalMaterial += (countSetBits(board.pieceBitboards[pt_base_idx]) + countSetBits(board.pieceBitboards[pt_base_idx + NUM_BASE_PIECE_TYPES])) * pieceMaterialValue[pt_base_idx];
    }
    // NOVO: Fase de Jogo - interpolação linear entre abertura e final
    int gamePhase = std::min(24, totalMaterial / (2 * (4*KNIGHT_VALUE + 4*BISHOP_VALUE + 4*ROOK_VALUE + 2*QUEEN_VALUE) / 100)); // Simples

    for (int pt_base_idx = 0; pt_base_idx < NUM_BASE_PIECE_TYPES; ++pt_base_idx) {
        PieceType whitePiece = static_cast<PieceType>(pt_base_idx);
        PieceType blackPiece = static_cast<PieceType>(pt_base_idx + NUM_BASE_PIECE_TYPES);

        materialScore += pieceMaterialValue[pt_base_idx] * (countSetBits(board.pieceBitboards[whitePiece]) - countSetBits(board.pieceBitboards[blackPiece]));
        
        Square sq;
        Bitboard temp_bb_white = board.pieceBitboards[whitePiece];
        while((sq = pop_lsb(temp_bb_white)) != NO_SQ) {
            int openingScore = pieceSquareTables_Opening[pt_base_idx][sq];
            int endgameScore = pieceSquareTables_Endgame[pt_base_idx][sq];
            positionalScore += ((openingScore * (24 - gamePhase)) + (endgameScore * gamePhase)) / 24;
        }

        Bitboard temp_bb_black = board.pieceBitboards[blackPiece];
        while((sq = pop_lsb(temp_bb_black)) != NO_SQ) {
            int openingScore = pieceSquareTables_Opening[pt_base_idx][mirror_square(sq)];
            int endgameScore = pieceSquareTables_Endgame[pt_base_idx][mirror_square(sq)];
            positionalScore -= ((openingScore * (24 - gamePhase)) + (endgameScore * gamePhase)) / 24;
        }
    }

    // --- PLACEHOLDERS PARA AVALIAÇÕES FUTURAS ---
    // score += evaluate_pawn_structure(board);
    // score += evaluate_king_safety(board);
    // score += evaluate_mobility(board);

    int score = materialScore + positionalScore;
    // Retorna a pontuação da perspectiva do jogador atual
    return (board.sideToMove == WHITE) ? score : -score;
}

// --- NOVO: Ordenação de Lances ---
void order_moves(const BoardState& board, std::vector<Move>& moves, const Move& ttMove, int ply) {
    const int MVV_LVA_SCORES[NUM_BASE_PIECE_TYPES][NUM_BASE_PIECE_TYPES] = {
      // Vítima:   P,  N,  B,  R,  Q,  K (Atacante)
      /* P */ { 105, 205, 305, 405, 505, 605 },
      /* N */ { 104, 204, 304, 404, 504, 604 },
      /* B */ { 103, 203, 303, 403, 503, 603 },
      /* R */ { 102, 202, 302, 402, 502, 602 },
      /* Q */ { 101, 201, 301, 401, 501, 601 },
      /* K */ { 100, 200, 300, 400, 500, 600 }
    };
    const int TT_MOVE_SCORE = 1000000;
    const int CAPTURE_SCORE = 800000;
    const int KILLER_SCORE = 600000;

    std::vector<ScoredMove> scoredMoves;
    for (const auto& move : moves) {
        int score = 0;
        if (ttMove.fromSquare != NO_SQ && move == ttMove) {
            score = TT_MOVE_SCORE;
        } else if (move.isCapture) {
            PieceType attacker = static_cast<PieceType>(move.pieceMoved % NUM_BASE_PIECE_TYPES);
            PieceType victim = static_cast<PieceType>(move.pieceCaptured % NUM_BASE_PIECE_TYPES);
            score = CAPTURE_SCORE + MVV_LVA_SCORES[victim][attacker];
        } else {
            if (killerMoves[ply][0] == move || killerMoves[ply][1] == move) {
                score = KILLER_SCORE;
            } else {
                score = historyScores[move.pieceMoved][move.toSquare];
            }
        }
        scoredMoves.emplace_back(move, score);
    }
    
    std::sort(scoredMoves.begin(), scoredMoves.end());

    moves.clear();
    for (const auto& scored_move : scoredMoves) {
        moves.push_back(scored_move.move);
    }
}

int quiescence_search(BoardState currentBoard, int alpha, int beta, int ply) {
    if (ply >= MAX_PLY) return evaluate(currentBoard);

    int stand_pat = evaluate(currentBoard);
    if (stand_pat >= beta) {
        return beta;
    }
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }

    std::vector<Move> captureMoves;
    generate_legal_moves(currentBoard, captureMoves, true);
    order_moves(currentBoard, captureMoves, Move(), ply); // Ordena as capturas com MVV-LVA

    for (const auto& move : captureMoves) {
        BoardState nextBoard = currentBoard;
        make_move(nextBoard, move, true);
        int score = -quiescence_search(nextBoard, -beta, -alpha, ply + 1);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}


int search(BoardState currentBoard, int depth, int alpha, int beta, int ply, bool isPV, bool isNullMove) {
    if (ply >= MAX_PLY) return evaluate(currentBoard);
    
    // --- Lógica de Empate na Busca ---
    if (currentBoard.halfMoveClock >= 100 || is_draw_by_repetition(currentBoard)) {
        return DRAW_SCORE;
    }

    // --- Atingiu a Profundidade Limite ---
    if (depth <= 0) {
        return quiescence_search(currentBoard, alpha, beta, ply);
    }

    bool isRoot = (ply == 0);
    // --- Probe na Tabela de Transposição ---
    Bitboard key = currentBoard.zobristKey;
    TTEntry& tt_entry = transpositionTable[key & (TT_SIZE - 1)];
    if (!isRoot && tt_entry.key == key && tt_entry.depth >= depth) {
        if (tt_entry.flag == TT_EXACT) return tt_entry.score;
        if (tt_entry.flag == TT_LOWER_BOUND && tt_entry.score >= beta) return beta;
        if (tt_entry.flag == TT_UPPER_BOUND && tt_entry.score <= alpha) return alpha;
    }

    bool inCheck = is_king_in_check(currentBoard, currentBoard.sideToMove);
    if (inCheck) depth++; // Extensão de busca em xeque

    // --- Null Move Pruning (NMP) ---
    if (!isPV && !inCheck && !isNullMove && depth >= 3) {
        BoardState nullMoveBoard = currentBoard;
        nullMoveBoard.sideToMove = (currentBoard.sideToMove == WHITE) ? BLACK : WHITE;
        nullMoveBoard.zobristKey ^= zobristSideToMoveKey;

        int null_score = -search(nullMoveBoard, depth - 1 - 2, -beta, -beta + 1, ply + 1, false, true);
        if (null_score >= beta) {
            return beta;
        }
    }

    // --- Início da Busca ---
    std::vector<Move> legalMoves;
    generate_legal_moves(currentBoard, legalMoves, false);
    
    if (legalMoves.empty()) {
        return inCheck ? -(MATE_SCORE - ply) : DRAW_SCORE; // Cheque-mate ou Afogamento
    }

    order_moves(currentBoard, legalMoves, tt_entry.bestMove, ply);

    int movesSearched = 0;
    int bestScore = -INFINITE_SCORE;
    Move bestMove;
    int original_alpha = alpha;

    for (const auto& move : legalMoves) {
        BoardState nextBoard = currentBoard;
        make_move(nextBoard, move, true);
        
        int score;
        if (movesSearched == 0) { // Primeiro lance: Busca PVS com janela completa
            score = -search(nextBoard, depth - 1, -beta, -alpha, ply + 1, true, false);
        } else { // Demais lances: Busca com Janela Nula
            score = -search(nextBoard, depth - 1, -alpha - 1, -alpha, ply + 1, false, false);
            if (score > alpha && score < beta) { // Se falhar, re-busca com janela completa
                score = -search(nextBoard, depth - 1, -beta, -alpha, ply + 1, true, false);
            }
        }
        
        movesSearched++;

        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
            if (score > alpha) {
                alpha = score;
                if (score >= beta) { // Beta-cutoff
                    if (!move.isCapture) {
                        killerMoves[ply][1] = killerMoves[ply][0];
                        killerMoves[ply][0] = move;
                        historyScores[move.pieceMoved][move.toSquare] += depth * depth;
                    }
                    goto store_tt; // Pula para o armazenamento na TT
                }
            }
        }
    }

store_tt:
    tt_entry.key = key;
    tt_entry.depth = depth;
    tt_entry.score = bestScore;
    tt_entry.bestMove = bestMove;
    if (bestScore <= original_alpha) tt_entry.flag = TT_UPPER_BOUND;
    else if (bestScore >= beta) tt_entry.flag = TT_LOWER_BOUND;
    else tt_entry.flag = TT_EXACT;

    return bestScore;
}


Move find_best_move(BoardState& board, int maxDepth, int timeLimitMs) {
    clear_search_heuristics();
    gameHistory.clear(); // Limpa histórico para nova busca

    Move bestMove;
    int bestScore = -INFINITE_SCORE;
    auto t_start = std::chrono::high_resolution_clock::now();

    // --- NOVO: Aprofundamento Iterativo (Iterative Deepening) ---
    for (int currentDepth = 1; currentDepth <= maxDepth; ++currentDepth) {
        int score = search(board, currentDepth, -INFINITE_SCORE, INFINITE_SCORE, 0, true, false);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

        // Obtem o melhor lance da TT
        TTEntry& tt_entry = transpositionTable[board.zobristKey & (TT_SIZE - 1)];
        if (tt_entry.key == board.zobristKey) {
            bestMove = tt_entry.bestMove;
            bestScore = tt_entry.score;
        }

        std::cout << "info depth " << currentDepth << " score cp " << bestScore << " time " << static_cast<int>(elapsed_ms)
                  << " pv " << bestMove.toString() << std::endl;
        
        // Condição de parada por tempo
        if (elapsed_ms > timeLimitMs) {
            std::cout << "Tempo limite atingido, usando melhor lance da profundidade " << currentDepth << std::endl;
            break;
        }
    }
    
    std::cout << "\nMelhor lance final: " << bestMove.toString() << " com pontuacao " << bestScore << std::endl;
    return bestMove;
}


int main() {
    initialize_all();
    BoardState board;
    initialize_board(board);

    std::string userInput;
    Color playerColor = WHITE;
    int searchDepth = 10; // Profundidade máxima para a busca iterativa
    int timePerMove = 5000; // 5 segundos por lance

    while(true) {
        print_pretty_board(board);
        std::cout << "Avaliacao Estatica (Visao Brancas): " << (board.sideToMove == WHITE ? evaluate(board) : -evaluate(board)) << std::endl;

        std::vector<Move> legal_moves;
        generate_legal_moves(board, legal_moves, false);

        if (legal_moves.empty()) {
            if (is_king_in_check(board, board.sideToMove)) {
                std::cout << "XEQUE-MATE! As " << (board.sideToMove == WHITE ? "Pretas" : "Brancas") << " venceram!" << std::endl;
            } else {
                std::cout << "AFOGAMENTO! Empate." << std::endl;
            }
            break;
        }
        if (board.halfMoveClock >= 100 || is_draw_by_repetition(board)) {
            std::cout << "EMPATE por regra dos 50 lances ou por repeticao!" << std::endl;
            break;
        }
        
        Move chosenMove;
        if (board.sideToMove == playerColor) {
            std::cout << "Seu lance (ex: e2e4) ou 'quit': ";
            std::cin >> userInput;
            if (userInput == "quit") break;
            chosenMove = parse_move_input(userInput, legal_moves, board.sideToMove);
            if (chosenMove.fromSquare == NO_SQ) {
                std::cout << "Lance invalido. Tente novamente." << std::endl;
                continue;
            }
        } else {
            chosenMove = find_best_move(board, searchDepth, timePerMove);
            if (chosenMove.fromSquare == NO_SQ) {
                std::cout << "Motor nao encontrou um lance!" << std::endl;
                break;
            }
        }
        
        make_move(board, chosenMove); // Usa a função principal que atualiza o histórico
    }
    
    return 0;
}


// --- Funções Utilitárias e de Inicialização (sem alterações significativas) ---
// (As implementações completas de initialize_board, print_pretty_board, parse_move_input, e os geradores de lances
// pseudo-legais são necessárias aqui, mas foram omitidas na exibição para focar nas melhorias.)

// Exemplo de como as funções omitidas devem ser incluídas:
void initialize_board(BoardState &board) {
    for (int i = 0; i < NUM_PIECE_TYPES; ++i) board.pieceBitboards[i] = 0ULL;
    board.colorBitboards[WHITE] = 0ULL; board.colorBitboards[BLACK] = 0ULL;
    board.occupiedBitboard = 0ULL;

    board.pieceBitboards[WP] = RANK_2_BB;
    board.pieceBitboards[BP] = RANK_7_BB;

    set_bit(board.pieceBitboards[WR], A1); set_bit(board.pieceBitboards[WR], H1);
    set_bit(board.pieceBitboards[WN], B1); set_bit(board.pieceBitboards[WN], G1);
    set_bit(board.pieceBitboards[WB], C1); set_bit(board.pieceBitboards[WB], F1);
    set_bit(board.pieceBitboards[WQ], D1); set_bit(board.pieceBitboards[WK], E1);

    set_bit(board.pieceBitboards[BR], A8); set_bit(board.pieceBitboards[BR], H8);
    set_bit(board.pieceBitboards[BN], B8); set_bit(board.pieceBitboards[BN], G8);
    set_bit(board.pieceBitboards[BB], C8); set_bit(board.pieceBitboards[BB], F8);
    set_bit(board.pieceBitboards[BQ], D8); set_bit(board.pieceBitboards[BK], E8);

    for (int pt_idx = WP; pt_idx <= WK; ++pt_idx) board.colorBitboards[WHITE] |= board.pieceBitboards[pt_idx];
    for (int pt_idx = BP; pt_idx <= BK; ++pt_idx) board.colorBitboards[BLACK] |= board.pieceBitboards[pt_idx];
    board.occupiedBitboard = board.colorBitboards[WHITE] | board.colorBitboards[BLACK];

    board.sideToMove = WHITE;
    board.castlingRights = WK_CASTLE | WQ_CASTLE | BK_CASTLE | BQ_CASTLE;
    board.enPassantSquare = NO_SQ;
    board.halfMoveClock = 0;
    board.fullMoveNumber = 1;

    board.zobristKey = generate_zobrist_key(board);
}

void print_pretty_board(const BoardState &board) {
    std::cout << std::endl << "  +---+---+---+---+---+---+---+---+" << std::endl;
    for (int rank = 7; rank >= 0; --rank) {
        std::cout << rank + 1 << " |";
        for (int file = 0; file <= 7; ++file) {
            Square sq = static_cast<Square>(rank * 8 + file);
            PieceType currentPiece = NO_PIECE;
            for (int pt_idx = 0; pt_idx < NUM_PIECE_TYPES; ++pt_idx) {
                if (get_bit(board.pieceBitboards[pt_idx], sq)) {
                    currentPiece = static_cast<PieceType>(pt_idx); break;
                }
            }
            std::cout << " " << piece_to_char(currentPiece) << " |";
        }
        std::cout << std::endl << "  +---+---+---+---+---+---+---+---+" << std::endl;
    }
    std::cout << "    a   b   c   d   e   f   g   h" << std::endl << std::endl;
    std::cout << "Turno: " << (board.sideToMove == WHITE ? "Brancas" : "Pretas") << std::endl;
    if (is_king_in_check(board, board.sideToMove)) {
        std::cout << "STATUS: XEQUE!" << std::endl;
    }
    std::cout << "Roque: ";
    if (board.castlingRights & WK_CASTLE) std::cout << "K"; if (board.castlingRights & WQ_CASTLE) std::cout << "Q";
    if (board.castlingRights & BK_CASTLE) std::cout << "k"; if (board.castlingRights & BQ_CASTLE) std::cout << "q";
    if (board.castlingRights == 0) std::cout << "-";
    std::cout << std::endl;
    std::cout << "En Passant: ";
    if (board.enPassantSquare != NO_SQ) {
        std::cout << static_cast<char>('a' + (board.enPassantSquare % 8)) << static_cast<char>('1' + (board.enPassantSquare / 8));
    } else { std::cout << "-"; }
    std::cout << std::endl;
    std::cout << "Meio-lance: " << board.halfMoveClock << ", Lance: " << board.fullMoveNumber << std::endl;
}

Move parse_move_input(const std::string& moveStr, const std::vector<Move>& legalMoves, Color currentSide) {
    if (moveStr.length() < 4 || moveStr.length() > 5) return Move();
    Square fromSq = static_cast<Square>((moveStr[0] - 'a') + (moveStr[1] - '1') * 8);
    Square toSq = static_cast<Square>((moveStr[2] - 'a') + (moveStr[3] - '1') * 8);
    PieceType promotedPiece = NO_PIECE;
    if (moveStr.length() == 5) {
        char promoChar = moveStr[4];
        if (currentSide == WHITE) {
            if (promoChar == 'q') promotedPiece = WQ; else if (promoChar == 'r') promotedPiece = WR;
            else if (promoChar == 'b') promotedPiece = WB; else if (promoChar == 'n') promotedPiece = WN;
            else return Move(); 
        } else { 
            if (promoChar == 'q') promotedPiece = BQ; else if (promoChar == 'r') promotedPiece = BR;
            else if (promoChar == 'b') promotedPiece = BB; else if (promoChar == 'n') promotedPiece = BN;
            else return Move(); 
        }
    }
    for (const auto& legalMove : legalMoves) {
        if (legalMove.fromSquare == fromSq && legalMove.toSquare == toSq) {
            if (legalMove.isPromotion) { if (legalMove.promotedTo == promotedPiece) return legalMove; } 
            else return legalMove;
        }
    }
    return Move(); 
}
