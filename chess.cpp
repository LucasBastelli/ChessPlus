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

// === Constantes e Estruturas Globais ===

using Bitboard = unsigned long long;

// Constantes para as peças e cores
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

// Constantes de Jogo e Busca
const int WK_CASTLE = 1, WQ_CASTLE = 2, BK_CASTLE = 4, BQ_CASTLE = 8; 
const int PAWN_VALUE = 100, KNIGHT_VALUE = 320, BISHOP_VALUE = 330, ROOK_VALUE = 500, QUEEN_VALUE = 900, KING_VALUE_MATERIAL = 0;
const int MATE_SCORE = 20000, INFINITE_SCORE = MATE_SCORE * 2, DEFAULT_SEARCH_DEPTH = 7, MAX_QUIESCENCE_PLY = 5;
const int CONTEMPT_FACTOR = 30; // discourages draws when ahead

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

std::array<int, NUM_BASE_PIECE_TYPES> pieceMaterialValue = { PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE_MATERIAL };
std::array<std::array<int, 64>, NUM_BASE_PIECE_TYPES> pieceSquareTables;
std::array<std::array<int, NUM_BASE_PIECE_TYPES>, NUM_BASE_PIECE_TYPES> mvv_lva;
std::array<Bitboard, 64> knightAttacks;
std::array<Bitboard, 64> kingAttacks;

// --- Zobrist Hashing e Tabela de Transposição ---
std::array<std::array<Bitboard, 64>, NUM_PIECE_TYPES> zobristPieceKeys;
Bitboard zobristSideToMoveKey;
std::array<Bitboard, 16> zobristCastlingKeys; 
std::array<Bitboard, 64> zobristEnPassantKeys;

enum TT_FLAG { TT_EXACT, TT_LOWER_BOUND, TT_UPPER_BOUND };
struct TTEntry {
    Bitboard key;       
    int depth;          
    int score;          
    TT_FLAG flag;       
};

const int TT_SIZE = 1048576; 
std::vector<TTEntry> transpositionTable(TT_SIZE);

// --- Estruturas de Lance ---
struct Move {
    Square fromSquare; Square toSquare; PieceType pieceMoved; PieceType pieceCaptured; PieceType promotedTo;    
    bool isCapture; bool isEnPassant; bool isPromotion; bool isCastleKingside; bool isCastleQueenside;
    Move() : fromSquare(NO_SQ),pieceMoved(NO_PIECE){} 
    Move(Square from, Square to, PieceType moved, PieceType captured = NO_PIECE, PieceType promoted = NO_PIECE,
         bool ep = false, bool castleK = false, bool castleQ = false)
        : fromSquare(from), toSquare(to), pieceMoved(moved), pieceCaptured(captured),
          promotedTo(promoted), isCapture(captured != NO_PIECE || ep), isEnPassant(ep), 
          isPromotion(promoted != NO_PIECE), isCastleKingside(castleK), isCastleQueenside(castleQ) {}
    std::string toString() const; 
    bool operator==(const Move& other) const; 
};

struct ScoredMove {
    Move move;
    int score;
    ScoredMove(Move m, int s) : move(m), score(s) {}
    bool operator<(const ScoredMove& other) const { return score > other.score; }
};

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
#warning "Using fallback LSB and popcount functions, performance will be SEVERELY impacted."
inline int countSetBits_fallback(Bitboard bb) { int count = 0; while (bb > 0) { bb &= (bb - 1); count++; } return count; }
inline int get_lsb_index_fallback(Bitboard bb) { if (bb == 0) return -1; for (int i = 0; i < 64; i++) { if ((bb >> i) & 1) return i; } return -1;}
inline Square pop_lsb_fallback(Bitboard &bb) { if (bb == 0) return NO_SQ; Square lsb_sq = static_cast<Square>(get_lsb_index_fallback(bb)); if (lsb_sq != NO_SQ) { clear_bit(bb, lsb_sq); } return lsb_sq;}
inline int get_lsb_index(Bitboard bb) { return get_lsb_index_fallback(bb); }
inline Square pop_lsb(Bitboard &bb) { return pop_lsb_fallback(bb); }
inline int countSetBits(Bitboard bb) { return countSetBits_fallback(bb); }
#endif

inline Square mirror_square(Square sq) { return static_cast<Square>(sq ^ 56); }


// --- Protótipos de Funções (para o compilador saber que elas existem) ---
void initialize_zobrist_keys();
Bitboard generate_zobrist_key(const BoardState& board);
void clear_transposition_table();
bool is_square_attacked(Square sq, Color attackerColor, const BoardState &board);
void make_move_internal(BoardState &board, const Move &move); 
bool is_king_in_check(const BoardState &board, Color kingColor); 
void generate_pawn_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_knight_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_bishop_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_rook_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_queen_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_king_pseudo_moves(const BoardState &board, std::vector<Move> &moveList);
void generate_sliding_pseudo_moves(const BoardState &board, std::vector<Move> &moveList, PieceType pt, const int directions[], int num_directions);
void generate_legal_moves(const BoardState &original_board, std::vector<Move> &legalMoveList, bool capturesOnly);
int evaluate_king_safety(const BoardState &board);
int evaluate_pawn_structure(const BoardState &board);
int evaluate_mobility(const BoardState &board);
int evaluate(const BoardState &board);
int quiescence_search(BoardState currentBoard, int alpha, int beta, int q_depth);
int negamax_search(BoardState currentBoard, int depth, int alpha, int beta, std::vector<Bitboard>& history);
Move find_best_move(const BoardState& board, int searchDepth, const std::vector<Bitboard>& history);
void initialize_board(BoardState &board);
void print_pretty_board(const BoardState &board);
Move parse_move_input(const std::string& moveStr, const std::vector<Move>& legalMoves, Color currentSide);
void initialize_attack_tables();
void initialize_evaluation_parameters();
void initialize_mvv_lva();
char piece_to_char(PieceType pt);
int search_root_move_task(BoardState boardAfterMove, int searchDepth, std::vector<Bitboard> history);
int count_repetitions(const std::vector<Bitboard>& history, Bitboard key);


// --- Implementações Completas das Funções ---

std::string Move::toString() const {
    if (fromSquare == NO_SQ) return "invalid_move";
    std::string moveStr;
    moveStr += static_cast<char>('a' + (fromSquare % 8));
    moveStr += static_cast<char>('1' + (fromSquare / 8));
    moveStr += static_cast<char>('a' + (toSquare % 8));
    moveStr += static_cast<char>('1' + (toSquare / 8));
    if (isPromotion) {
        char promoChar = ' ';
        PieceType basePromoPiece = promotedTo;
        if (promotedTo >= BP && promotedTo <= BK) { 
            basePromoPiece = static_cast<PieceType>(promotedTo - (BP - WP));
        }
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
    return fromSquare == other.fromSquare && toSquare == other.toSquare &&
           pieceMoved == other.pieceMoved && promotedTo == other.promotedTo;
}

void clear_transposition_table() {
    for(int i = 0; i < TT_SIZE; ++i) {
        transpositionTable[i] = {0, 0, 0, TT_EXACT};
    }
}

int count_repetitions(const std::vector<Bitboard>& history, Bitboard key) {
    return std::count(history.begin(), history.end(), key);
}

void initialize_zobrist_keys() {
    std::mt19937_64 randomEngine(123456789); 
    std::uniform_int_distribution<Bitboard> dist(0, std::numeric_limits<Bitboard>::max());

    for (int piece = 0; piece < NUM_PIECE_TYPES; ++piece) {
        for (int square = 0; square < 64; ++square) {
            zobristPieceKeys[piece][square] = dist(randomEngine);
        }
    }
    zobristSideToMoveKey = dist(randomEngine);
    for (int i = 0; i < 16; ++i) {
        zobristCastlingKeys[i] = dist(randomEngine);
    }
    for (int i = 0; i < 64; ++i) {
        zobristEnPassantKeys[i] = dist(randomEngine);
    }
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
    if (board.sideToMove == BLACK) {
        key ^= zobristSideToMoveKey;
    }
    key ^= zobristCastlingKeys[board.castlingRights];
    if (board.enPassantSquare != NO_SQ) {
        key ^= zobristEnPassantKeys[board.enPassantSquare];
    }
    return key;
}

void initialize_evaluation_parameters() {
    pieceSquareTables[WP] = {
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    };
    pieceSquareTables[WN] = {
        -50,-40,-30,-30,-30,-30,-40,-50, -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30, -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30, -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50
    };
    pieceSquareTables[WB] = {
        -20,-10,-10,-10,-10,-10,-10,-20, -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10, -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10, -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10, -20,-10,-10,-10,-10,-10,-10,-20
    };
    pieceSquareTables[WR] = {
         0,  0,  0,  0,  0,  0,  0,  0,   5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,  -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,  -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,   0,  0,  0,  5,  5,  0,  0,  0
    };
    pieceSquareTables[WQ] = {
        -20,-10,-10, -5, -5,-10,-10,-20, -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,  -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5, -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10, -20,-10,-10, -5, -5,-10,-10,-20
    };
    pieceSquareTables[WK] = {
        -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,  20, 30, 10,  0,  0, 10, 30, 20
    };
}

void initialize_mvv_lva() {
    for (int victim = 0; victim < NUM_BASE_PIECE_TYPES; ++victim) {
        for (int attacker = 0; attacker < NUM_BASE_PIECE_TYPES; ++attacker) {
            mvv_lva[victim][attacker] = pieceMaterialValue[victim] * 10 - pieceMaterialValue[attacker];
        }
    }
}

void initialize_attack_tables() { 
    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        Bitboard spot = 1ULL << sq_idx;
        knightAttacks[sq_idx] = 0ULL;
        knightAttacks[sq_idx] |= (spot << 17) & NOT_A_FILE; 
        knightAttacks[sq_idx] |= (spot << 15) & NOT_H_FILE; 
        knightAttacks[sq_idx] |= (spot << 10) & NOT_AB_FILE;
        knightAttacks[sq_idx] |= (spot <<  6) & NOT_HG_FILE;
        knightAttacks[sq_idx] |= (spot >> 17) & NOT_H_FILE; 
        knightAttacks[sq_idx] |= (spot >> 15) & NOT_A_FILE; 
        knightAttacks[sq_idx] |= (spot >> 10) & NOT_HG_FILE;
        knightAttacks[sq_idx] |= (spot >>  6) & NOT_AB_FILE;

        kingAttacks[sq_idx] = 0ULL;
        kingAttacks[sq_idx] |= (spot << 1) & NOT_A_FILE; 
        kingAttacks[sq_idx] |= (spot >> 1) & NOT_H_FILE; 
        kingAttacks[sq_idx] |= (spot << 8);             
        kingAttacks[sq_idx] |= (spot >> 8);             
        kingAttacks[sq_idx] |= (spot << 9) & NOT_A_FILE; 
        kingAttacks[sq_idx] |= (spot << 7) & NOT_H_FILE; 
        kingAttacks[sq_idx] |= (spot >> 7) & NOT_A_FILE; 
        kingAttacks[sq_idx] |= (spot >> 9) & NOT_H_FILE; 
    }
}
char piece_to_char(PieceType pt) { 
    switch (pt) {
        case WP: return 'P'; case WN: return 'N'; case WB: return 'B'; case WR: return 'R'; case WQ: return 'Q'; case WK: return 'K';
        case BP: return 'p'; case BN: return 'n'; case BB: return 'b'; case BR: return 'r'; case BQ: return 'q'; case BK: return 'k';
        default: return ' ';
    }
}

void make_move_internal(BoardState &board, const Move &move) {
    int old_castling_rights = board.castlingRights;
    board.zobristKey ^= zobristCastlingKeys[old_castling_rights];
    if (board.enPassantSquare != NO_SQ) {
        board.zobristKey ^= zobristEnPassantKeys[board.enPassantSquare];
    }
    
    Color us = board.sideToMove;
    Color them = (us == WHITE) ? BLACK : WHITE;

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
    
    if (move.pieceMoved == WK) { board.castlingRights &= ~(WK_CASTLE | WQ_CASTLE); }
    if (move.pieceMoved == BK) { board.castlingRights &= ~(BK_CASTLE | BQ_CASTLE); }
    if (move.fromSquare == H1 || move.toSquare == H1) { board.castlingRights &= ~WK_CASTLE; }
    if (move.fromSquare == A1 || move.toSquare == A1) { board.castlingRights &= ~WQ_CASTLE; }
    if (move.fromSquare == H8 || move.toSquare == H8) { board.castlingRights &= ~BK_CASTLE; }
    if (move.fromSquare == A8 || move.toSquare == A8) { board.castlingRights &= ~BQ_CASTLE; }
    if (move.pieceCaptured == WR && move.toSquare == H1) { board.castlingRights &= ~WK_CASTLE; }
    if (move.pieceCaptured == WR && move.toSquare == A1) { board.castlingRights &= ~WQ_CASTLE; }
    if (move.pieceCaptured == BR && move.toSquare == H8) { board.castlingRights &= ~BK_CASTLE; }
    if (move.pieceCaptured == BR && move.toSquare == A8) { board.castlingRights &= ~BQ_CASTLE; }

    board.enPassantSquare = NO_SQ; 
    if (move.pieceMoved == WP && move.toSquare - move.fromSquare == 16) { 
        board.enPassantSquare = static_cast<Square>(move.fromSquare + 8);
    } else if (move.pieceMoved == BP && move.fromSquare - move.toSquare == 16) { 
        board.enPassantSquare = static_cast<Square>(move.fromSquare - 8);
    }

    board.zobristKey ^= zobristCastlingKeys[board.castlingRights];
    if (board.enPassantSquare != NO_SQ) {
        board.zobristKey ^= zobristEnPassantKeys[board.enPassantSquare];
    }
    board.zobristKey ^= zobristSideToMoveKey;

    if (move.pieceMoved == WP || move.pieceMoved == BP || move.isCapture) {
        board.halfMoveClock = 0;
    } else {
        board.halfMoveClock++;
    }
    if (us == BLACK) {
        board.fullMoveNumber++;
    }
    board.sideToMove = them;
    board.occupiedBitboard = board.colorBitboards[WHITE] | board.colorBitboards[BLACK];
}

bool is_king_in_check(const BoardState &board, Color kingColor) {
    PieceType kingPiece = (kingColor == WHITE) ? WK : BK;
    Bitboard kingBb = board.pieceBitboards[kingPiece];
    if (kingBb == 0) return false; 
    
    Square kingSq = static_cast<Square>(get_lsb_index(kingBb));
    Color attackerColor = (kingColor == WHITE) ? BLACK : WHITE;
    
    return is_square_attacked(kingSq, attackerColor, board);
}

void generate_pawn_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) {
    Color us = board.sideToMove;
    Color them = (us == WHITE) ? BLACK : WHITE;
    PieceType ourPawn = (us == WHITE) ? WP : BP;
    PieceType enemyPawnForEP = (us == WHITE) ? BP : WP; 
    Bitboard pawns = board.pieceBitboards[ourPawn];
    Bitboard enemyPieces = board.colorBitboards[them];
    Bitboard allPieces = board.occupiedBitboard;
    Square fromSq, toSq;
    if (us == WHITE) {
        Bitboard push1_targets = (pawns << 8) & ~allPieces;
        Bitboard push2_targets = ((push1_targets & RANK_3_BB) << 8) & ~allPieces; 
        Bitboard capture_left_targets = ((pawns & NOT_A_FILE) << 7) & enemyPieces; 
        Bitboard capture_right_targets = ((pawns & NOT_H_FILE) << 9) & enemyPieces;
        PieceType promotionPieces[] = {WQ, WR, WB, WN}; 
        Bitboard temp_push1 = push1_targets;
        while((toSq = pop_lsb(temp_push1)) != NO_SQ) {
            fromSq = static_cast<Square>(toSq - 8);
            if (get_bit(RANK_8_BB, toSq)) { 
                for (PieceType promo_p : promotionPieces) moveList.emplace_back(fromSq, toSq, ourPawn, NO_PIECE, promo_p);
            } else { moveList.emplace_back(fromSq, toSq, ourPawn); }
        }
        Bitboard temp_push2 = push2_targets;
        while((toSq = pop_lsb(temp_push2)) != NO_SQ) {
            fromSq = static_cast<Square>(toSq - 16); moveList.emplace_back(fromSq, toSq, ourPawn);
        }
        Bitboard temp_cap_left = capture_left_targets;
        while((toSq = pop_lsb(temp_cap_left)) != NO_SQ) {
            fromSq = static_cast<Square>(toSq - 7); PieceType captured = NO_PIECE; 
            for(int pt_idx = BP; pt_idx <= BK; ++pt_idx) { if(get_bit(board.pieceBitboards[pt_idx], toSq)) { captured = static_cast<PieceType>(pt_idx); break;}}
            if (get_bit(RANK_8_BB, toSq)) { 
                for (PieceType promo_p : promotionPieces) moveList.emplace_back(fromSq, toSq, ourPawn, captured, promo_p);
            } else { moveList.emplace_back(fromSq, toSq, ourPawn, captured); }
        }
        Bitboard temp_cap_right = capture_right_targets;
        while((toSq = pop_lsb(temp_cap_right)) != NO_SQ) {
            fromSq = static_cast<Square>(toSq - 9); PieceType captured = NO_PIECE;
            for(int pt_idx = BP; pt_idx <= BK; ++pt_idx) { if(get_bit(board.pieceBitboards[pt_idx], toSq)) { captured = static_cast<PieceType>(pt_idx); break;}}
            if (get_bit(RANK_8_BB, toSq)) { 
                for (PieceType promo_p : promotionPieces) moveList.emplace_back(fromSq, toSq, ourPawn, captured, promo_p);
            } else { moveList.emplace_back(fromSq, toSq, ourPawn, captured); }
        }
        if (board.enPassantSquare != NO_SQ) {
            Bitboard ep_capture_target_bb = 1ULL << board.enPassantSquare;
            Bitboard possible_attacker_left = (ep_capture_target_bb >> 7) & NOT_H_FILE; 
            if ( (possible_attacker_left & pawns & RANK_5_BB) != 0 ) { 
                 moveList.emplace_back(static_cast<Square>(get_lsb_index(possible_attacker_left)), board.enPassantSquare, ourPawn, enemyPawnForEP, NO_PIECE, true);
            }
            Bitboard possible_attacker_right = (ep_capture_target_bb >> 9) & NOT_A_FILE; 
            if ( (possible_attacker_right & pawns & RANK_5_BB) != 0 ) { 
                moveList.emplace_back(static_cast<Square>(get_lsb_index(possible_attacker_right)), board.enPassantSquare, ourPawn, enemyPawnForEP, NO_PIECE, true);
            }
        }
    } else { // BLACK's turn
        Bitboard push1_targets = (pawns >> 8) & ~allPieces;
        Bitboard push2_targets = ((push1_targets & RANK_6_BB) >> 8) & ~allPieces; 
        Bitboard capture_left_targets = ((pawns & NOT_A_FILE) >> 9) & enemyPieces; 
        Bitboard capture_right_targets = ((pawns & NOT_H_FILE) >> 7) & enemyPieces;
        PieceType promotionPieces[] = {BQ, BR, BB, BN}; 
        Bitboard temp_push1 = push1_targets;
        while((toSq = pop_lsb(temp_push1)) != NO_SQ) {
            fromSq = static_cast<Square>(toSq + 8);
            if (get_bit(RANK_1_BB, toSq)) {
                for (PieceType promo_p : promotionPieces) moveList.emplace_back(fromSq, toSq, ourPawn, NO_PIECE, promo_p);
            } else { moveList.emplace_back(fromSq, toSq, ourPawn); }
        }
        Bitboard temp_push2 = push2_targets;
        while((toSq = pop_lsb(temp_push2)) != NO_SQ) {
            fromSq = static_cast<Square>(toSq + 16); moveList.emplace_back(fromSq, toSq, ourPawn);
        }
        Bitboard temp_cap_left = capture_left_targets;
        while((toSq = pop_lsb(temp_cap_left)) != NO_SQ) {
            fromSq = static_cast<Square>(toSq + 9); PieceType captured = NO_PIECE;
            for(int pt_idx = WP; pt_idx <= WK; ++pt_idx) { if(get_bit(board.pieceBitboards[pt_idx], toSq)) { captured = static_cast<PieceType>(pt_idx); break;}}
            if (get_bit(RANK_1_BB, toSq)) {
                for (PieceType promo_p : promotionPieces) moveList.emplace_back(fromSq, toSq, ourPawn, captured, promo_p);
            } else { moveList.emplace_back(fromSq, toSq, ourPawn, captured); }
        }
        Bitboard temp_cap_right = capture_right_targets;
        while((toSq = pop_lsb(temp_cap_right)) != NO_SQ) {
            fromSq = static_cast<Square>(toSq + 7); PieceType captured = NO_PIECE;
            for(int pt_idx = WP; pt_idx <= WK; ++pt_idx) { if(get_bit(board.pieceBitboards[pt_idx], toSq)) { captured = static_cast<PieceType>(pt_idx); break;}}
            if (get_bit(RANK_1_BB, toSq)) {
                for (PieceType promo_p : promotionPieces) moveList.emplace_back(fromSq, toSq, ourPawn, captured, promo_p);
            } else { moveList.emplace_back(fromSq, toSq, ourPawn, captured); }
        }
        if (board.enPassantSquare != NO_SQ) {
            Bitboard ep_capture_target_bb = 1ULL << board.enPassantSquare;
            Bitboard possible_attacker_left = (ep_capture_target_bb << 7) & NOT_H_FILE;
            if ( (possible_attacker_left & pawns & RANK_4_BB) != 0 ) { 
                moveList.emplace_back(static_cast<Square>(get_lsb_index(possible_attacker_left)), board.enPassantSquare, ourPawn, enemyPawnForEP, NO_PIECE, true);
            }
            Bitboard possible_attacker_right = (ep_capture_target_bb << 9) & NOT_A_FILE;
            if ( (possible_attacker_right & pawns & RANK_4_BB) != 0 ) { 
                moveList.emplace_back(static_cast<Square>(get_lsb_index(possible_attacker_right)), board.enPassantSquare, ourPawn, enemyPawnForEP, NO_PIECE, true);
            }
        }
    }
}
void generate_knight_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) {
    Color us = board.sideToMove; PieceType ourKnight = (us == WHITE) ? WN : BN;
    Bitboard knights = board.pieceBitboards[ourKnight];
    Bitboard ourPieces = board.colorBitboards[us];
    Bitboard enemyPieces = board.colorBitboards[(us == WHITE) ? BLACK : WHITE]; Square fromSq;
    while ((fromSq = pop_lsb(knights)) != NO_SQ) {
        Bitboard attacks = knightAttacks[fromSq] & ~ourPieces; Square toSq;
        while ((toSq = pop_lsb(attacks)) != NO_SQ) {
            PieceType captured = NO_PIECE;
            if (get_bit(enemyPieces, toSq)) { 
                if (us == WHITE) { for(int pt_idx = BP; pt_idx <= BK; ++pt_idx) { if(get_bit(board.pieceBitboards[pt_idx], toSq)) { captured = static_cast<PieceType>(pt_idx); break;}}}
                else { for(int pt_idx = WP; pt_idx <= WK; ++pt_idx) { if(get_bit(board.pieceBitboards[pt_idx], toSq)) { captured = static_cast<PieceType>(pt_idx); break;}}}
            }
            moveList.emplace_back(fromSq, toSq, ourKnight, captured);
        }
    }
}
void generate_sliding_pseudo_moves(const BoardState &board, std::vector<Move> &moveList, PieceType pt, const int directions[], int num_directions) {
    Color us = board.sideToMove;
    Bitboard pieces = board.pieceBitboards[pt];
    Bitboard friendlyPieces = board.colorBitboards[us];
    Bitboard enemyPieces = board.colorBitboards[(us == WHITE) ? BLACK : WHITE];
    Square fromSq;
    while ((fromSq = pop_lsb(pieces)) != NO_SQ) {
        for (int i = 0; i < num_directions; ++i) {
            int direction = directions[i];
            Square currentSq = fromSq;
            while (true) {
                int nextSq_idx = currentSq + direction;
                if (nextSq_idx < 0 || nextSq_idx >= 64) break;
                if (abs((nextSq_idx % 8) - (currentSq % 8)) > 1) break;
                Square toSq = static_cast<Square>(nextSq_idx);
                if (get_bit(friendlyPieces, toSq)) break;
                PieceType captured = NO_PIECE;
                if (get_bit(enemyPieces, toSq)) {
                    if (us == WHITE) {
                        for(int pt_idx = BP; pt_idx <= BK; ++pt_idx) {
                            if(get_bit(board.pieceBitboards[pt_idx], toSq)) { captured = static_cast<PieceType>(pt_idx); break; }
                        }
                    } else {
                        for(int pt_idx = WP; pt_idx <= WK; ++pt_idx) {
                            if(get_bit(board.pieceBitboards[pt_idx], toSq)) { captured = static_cast<PieceType>(pt_idx); break; }
                        }
                    }
                    moveList.emplace_back(fromSq, toSq, pt, captured);
                    break;
                }
                moveList.emplace_back(fromSq, toSq, pt);
                currentSq = toSq;
            }
        }
    }
}
void generate_bishop_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) {
    PieceType ourBishop = (board.sideToMove == WHITE) ? WB : BB;
    const int bishop_directions[] = {-9, -7, 7, 9}; 
    generate_sliding_pseudo_moves(board, moveList, ourBishop, bishop_directions, 4);
}
void generate_rook_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) {
    PieceType ourRook = (board.sideToMove == WHITE) ? WR : BR;
    const int rook_directions[] = {-8, -1, 1, 8}; 
    generate_sliding_pseudo_moves(board, moveList, ourRook, rook_directions, 4);
}
void generate_queen_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) {
    PieceType ourQueen = (board.sideToMove == WHITE) ? WQ : BQ;
    const int queen_directions[] = {-9, -8, -7, -1, 1, 7, 8, 9}; 
    generate_sliding_pseudo_moves(board, moveList, ourQueen, queen_directions, 8);
}
void generate_king_pseudo_moves(const BoardState &board, std::vector<Move> &moveList) {
    Color us = board.sideToMove; PieceType ourKing = (us == WHITE) ? WK : BK;
    Bitboard king_bb = board.pieceBitboards[ourKing]; if (king_bb == 0) return; 
    Square fromSq = static_cast<Square>(get_lsb_index(king_bb)); 
    Bitboard ourPieces = board.colorBitboards[us];
    Bitboard enemyPieces = board.colorBitboards[(us == WHITE) ? BLACK : WHITE];
    Bitboard attacks = kingAttacks[fromSq] & ~ourPieces; Square toSq_mv;
    while ((toSq_mv = pop_lsb(attacks)) != NO_SQ) {
        PieceType captured = NO_PIECE;
        if (get_bit(enemyPieces, toSq_mv)) { 
            if (us == WHITE) { for(int pt_idx = BP; pt_idx <= BK; ++pt_idx) { if(get_bit(board.pieceBitboards[pt_idx], toSq_mv)) { captured = static_cast<PieceType>(pt_idx); break;}}}
            else { for(int pt_idx = WP; pt_idx <= WK; ++pt_idx) { if(get_bit(board.pieceBitboards[pt_idx], toSq_mv)) { captured = static_cast<PieceType>(pt_idx); break;}}}
        }
        moveList.emplace_back(fromSq, toSq_mv, ourKing, captured);
    }
    if (us == WHITE) {
        if ((board.castlingRights & WK_CASTLE) && !get_bit(board.occupiedBitboard, F1) && !get_bit(board.occupiedBitboard, G1) &&
            !is_square_attacked(E1, BLACK, board) && !is_square_attacked(F1, BLACK, board) && !is_square_attacked(G1, BLACK, board)) {
            moveList.emplace_back(E1, G1, WK, NO_PIECE, NO_PIECE, false, true, false); }
        if ((board.castlingRights & WQ_CASTLE) && !get_bit(board.occupiedBitboard, D1) && !get_bit(board.occupiedBitboard, C1) && !get_bit(board.occupiedBitboard, B1) &&
            !is_square_attacked(E1, BLACK, board) && !is_square_attacked(D1, BLACK, board) && !is_square_attacked(C1, BLACK, board)) {
            moveList.emplace_back(E1, C1, WK, NO_PIECE, NO_PIECE, false, false, true); }
    } else { 
        if ((board.castlingRights & BK_CASTLE) && !get_bit(board.occupiedBitboard, F8) && !get_bit(board.occupiedBitboard, G8) &&
            !is_square_attacked(E8, WHITE, board) && !is_square_attacked(F8, WHITE, board) && !is_square_attacked(G8, WHITE, board)) {
            moveList.emplace_back(E8, G8, BK, NO_PIECE, NO_PIECE, false, true, false); }
        if ((board.castlingRights & BQ_CASTLE) && !get_bit(board.occupiedBitboard, D8) && !get_bit(board.occupiedBitboard, C8) && !get_bit(board.occupiedBitboard, B8) &&
            !is_square_attacked(E8, WHITE, board) && !is_square_attacked(D8, WHITE, board) && !is_square_attacked(C8, WHITE, board)) {
            moveList.emplace_back(E8, C8, BK, NO_PIECE, NO_PIECE, false, false, true); }
    }
}
bool is_square_attacked(Square sq, Color attackerColor, const BoardState &board) {
    if (attackerColor == WHITE) { 
        if (!get_bit(FILE_A_BB, sq) && (sq >= 9) && get_bit(board.pieceBitboards[WP], static_cast<Square>(sq - 9))) return true; 
        if (!get_bit(FILE_H_BB, sq) && (sq >= 7) && get_bit(board.pieceBitboards[WP], static_cast<Square>(sq - 7))) return true; 
    } else { 
        if (!get_bit(FILE_A_BB, sq) && (sq <= H8 - 7) && get_bit(board.pieceBitboards[BP], static_cast<Square>(sq + 7))) return true; 
        if (!get_bit(FILE_H_BB, sq) && (sq <= H8 - 9) && get_bit(board.pieceBitboards[BP], static_cast<Square>(sq + 9))) return true; 
    }
    PieceType attackingKnight = (attackerColor == WHITE) ? WN : BN;
    if ((knightAttacks[sq] & board.pieceBitboards[attackingKnight]) != 0) return true; 
    PieceType attackingKing = (attackerColor == WHITE) ? WK : BK;
    if ((kingAttacks[sq] & board.pieceBitboards[attackingKing]) != 0) return true; 
    Bitboard occupied = board.occupiedBitboard;
    PieceType attackingBishop = (attackerColor == WHITE) ? WB : BB;
    PieceType attackingRook = (attackerColor == WHITE) ? WR : BR;
    PieceType attackingQueen = (attackerColor == WHITE) ? WQ : BQ;
    const int bishop_directions[] = {-9, -7, 7, 9};
    for (int dir_idx = 0; dir_idx < 4; ++dir_idx) {
        int direction = bishop_directions[dir_idx]; Square raySq = sq; 
        while(true) {
            int nextSq_idx = raySq + direction;
            if (nextSq_idx < 0 || nextSq_idx > 63) break;
            int current_file = raySq % 8; int next_file = nextSq_idx % 8;
            int current_rank = raySq / 8; int next_rank = nextSq_idx / 8;
            if (abs(next_file - current_file) != 1 || abs(next_rank - current_rank) != 1) break; 
            raySq = static_cast<Square>(nextSq_idx);
            if (get_bit(board.pieceBitboards[attackingBishop], raySq)) return true;
            if (get_bit(board.pieceBitboards[attackingQueen], raySq)) return true;
            if (get_bit(occupied, raySq)) break; 
        }
    }
    const int rook_directions[] = {-8, -1, 1, 8};
    for (int dir_idx = 0; dir_idx < 4; ++dir_idx) {
        int direction = rook_directions[dir_idx]; Square raySq = sq;
        while(true) {
            int nextSq_idx = raySq + direction;
            if (nextSq_idx < 0 || nextSq_idx > 63) break;
            int current_file = raySq % 8; int next_file = nextSq_idx % 8;
            int current_rank = raySq / 8; int next_rank = nextSq_idx / 8;
            if (abs(direction) == 1) { if (next_rank != current_rank) break; } 
            else { if (next_file != current_file && abs(direction) == 8 ) break; }
            raySq = static_cast<Square>(nextSq_idx);
            if (get_bit(board.pieceBitboards[attackingRook], raySq)) return true;
            if (get_bit(board.pieceBitboards[attackingQueen], raySq)) return true;
            if (get_bit(occupied, raySq)) break;
        }
    }
    return false;
}

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

void generate_legal_moves(const BoardState &original_board, std::vector<Move> &legalMoveList, bool capturesOnly) {
    legalMoveList.clear();
    std::vector<Move> pseudoLegalMoves;

    generate_pawn_pseudo_moves(original_board, pseudoLegalMoves);
    generate_knight_pseudo_moves(original_board, pseudoLegalMoves);
    generate_bishop_pseudo_moves(original_board, pseudoLegalMoves);
    generate_rook_pseudo_moves(original_board, pseudoLegalMoves);
    generate_queen_pseudo_moves(original_board, pseudoLegalMoves);
    generate_king_pseudo_moves(original_board, pseudoLegalMoves); 

    for (const auto& pseudoMove : pseudoLegalMoves) {
        if (capturesOnly && !pseudoMove.isCapture) {
            continue; 
        }
        BoardState tempBoard = original_board;
        make_move_internal(tempBoard, pseudoMove); 
        if (!is_king_in_check(tempBoard, original_board.sideToMove)) {
            legalMoveList.push_back(pseudoMove);
        }
    }
}

int evaluate_king_safety(const BoardState &board) {
    const int SHIELD_PENALTY = 20;
    int score = 0;

    int wKingIdx = get_lsb_index(board.pieceBitboards[WK]);
    if (wKingIdx != -1) {
        Square wKingSq = static_cast<Square>(wKingIdx);
        int shield = 0;
        if (static_cast<int>(wKingSq) <= 55) {
            if ((static_cast<int>(wKingSq) % 8) != 0 && get_bit(board.pieceBitboards[WP], static_cast<Square>(wKingSq + 7))) shield++;
            if (get_bit(board.pieceBitboards[WP], static_cast<Square>(wKingSq + 8))) shield++;
            if ((static_cast<int>(wKingSq) % 8) != 7 && get_bit(board.pieceBitboards[WP], static_cast<Square>(wKingSq + 9))) shield++;
        }
        score -= (3 - shield) * SHIELD_PENALTY;
    }

    int bKingIdx = get_lsb_index(board.pieceBitboards[BK]);
    if (bKingIdx != -1) {
        Square bKingSq = static_cast<Square>(bKingIdx);
        int shield = 0;
        if (static_cast<int>(bKingSq) >= 8) {
            if ((static_cast<int>(bKingSq) % 8) != 7 && get_bit(board.pieceBitboards[BP], static_cast<Square>(bKingSq - 7))) shield++;
            if (get_bit(board.pieceBitboards[BP], static_cast<Square>(bKingSq - 8))) shield++;
            if ((static_cast<int>(bKingSq) % 8) != 0 && get_bit(board.pieceBitboards[BP], static_cast<Square>(bKingSq - 9))) shield++;
        }
        score += (3 - shield) * SHIELD_PENALTY;
    }

    return score;
}

int evaluate_pawn_structure(const BoardState &board) {
    const int DOUBLED_PENALTY = 20;
    const int ISOLATED_PENALTY = 15;
    const int PASSED_BONUS[8] = {0,10,20,30,50,80,130,0};

    int score = 0;
    Bitboard whitePawns = board.pieceBitboards[WP];
    Bitboard blackPawns = board.pieceBitboards[BP];
    std::array<Bitboard,8> fileMasks = {FILE_A_BB,FILE_B_BB,FILE_C_BB,FILE_D_BB,FILE_E_BB,FILE_F_BB,FILE_G_BB,FILE_H_BB};

    for (int file = 0; file < 8; ++file) {
        Bitboard mask = fileMasks[file];
        int wCount = countSetBits(whitePawns & mask);
        int bCount = countSetBits(blackPawns & mask);
        if (wCount > 1) score -= (wCount - 1) * DOUBLED_PENALTY;
        if (bCount > 1) score += (bCount - 1) * DOUBLED_PENALTY;

        if (wCount > 0) {
            bool left = file > 0 && (whitePawns & fileMasks[file-1]);
            bool right = file < 7 && (whitePawns & fileMasks[file+1]);
            if (!left && !right) score -= ISOLATED_PENALTY * wCount;
        }
        if (bCount > 0) {
            bool left = file > 0 && (blackPawns & fileMasks[file-1]);
            bool right = file < 7 && (blackPawns & fileMasks[file+1]);
            if (!left && !right) score += ISOLATED_PENALTY * bCount;
        }
    }

    Bitboard wp = whitePawns; Square sq;
    while ((sq = pop_lsb(wp)) != NO_SQ) {
        int idx = static_cast<int>(sq);
        int file = idx % 8;
        int rank = idx / 8;
        bool blocked = false;
        for (int r = rank + 1; r < 8 && !blocked; ++r) {
            if (get_bit(blackPawns, static_cast<Square>(r*8 + file))) blocked = true;
            if (file > 0 && get_bit(blackPawns, static_cast<Square>(r*8 + file - 1))) blocked = true;
            if (file < 7 && get_bit(blackPawns, static_cast<Square>(r*8 + file + 1))) blocked = true;
        }
        if (!blocked) score += PASSED_BONUS[rank];
    }

    Bitboard bp = blackPawns;
    while ((sq = pop_lsb(bp)) != NO_SQ) {
        int idx = static_cast<int>(sq);
        int file = idx % 8;
        int rank = idx / 8;
        bool blocked = false;
        for (int r = rank - 1; r >= 0 && !blocked; --r) {
            if (get_bit(whitePawns, static_cast<Square>(r*8 + file))) blocked = true;
            if (file > 0 && get_bit(whitePawns, static_cast<Square>(r*8 + file - 1))) blocked = true;
            if (file < 7 && get_bit(whitePawns, static_cast<Square>(r*8 + file + 1))) blocked = true;
        }
        if (!blocked) score -= PASSED_BONUS[7 - rank];
    }

    return score;
}

int evaluate_mobility(const BoardState &board) {
    const int MOBILITY_WEIGHT = 2;
    BoardState temp = board;
    std::vector<Move> moves;

    temp.sideToMove = WHITE;
    generate_legal_moves(temp, moves, false);
    int whiteMobility = static_cast<int>(moves.size());

    temp.sideToMove = BLACK;
    generate_legal_moves(temp, moves, false);
    int blackMobility = static_cast<int>(moves.size());

    return (whiteMobility - blackMobility) * MOBILITY_WEIGHT;
}

int evaluate(const BoardState &board) {
    int score = 0;
    int materialScore = 0;
    int pstScore = 0;

    for (int pt_base_idx = 0; pt_base_idx < NUM_BASE_PIECE_TYPES; ++pt_base_idx) {
        PieceType whitePiece = static_cast<PieceType>(pt_base_idx); 
        PieceType blackPiece = static_cast<PieceType>(pt_base_idx + NUM_BASE_PIECE_TYPES); 

        Bitboard white_pieces_bb = board.pieceBitboards[whitePiece];
        Bitboard black_pieces_bb = board.pieceBitboards[blackPiece];

        materialScore += pieceMaterialValue[pt_base_idx] * countSetBits(white_pieces_bb);
        materialScore -= pieceMaterialValue[pt_base_idx] * countSetBits(black_pieces_bb);
        
        Square sq;
        Bitboard temp_bb_white = white_pieces_bb;
        while((sq = pop_lsb(temp_bb_white)) != NO_SQ) {
            pstScore += pieceSquareTables[pt_base_idx][sq];
        }

        Bitboard temp_bb_black = black_pieces_bb;
        while((sq = pop_lsb(temp_bb_black)) != NO_SQ) {
            pstScore -= pieceSquareTables[pt_base_idx][mirror_square(sq)];
        }
    }
    score = materialScore + pstScore;
    score += evaluate_king_safety(board);
    score += evaluate_pawn_structure(board);
    score += evaluate_mobility(board);

    const int BISHOP_PAIR_BONUS = 40;
    if (countSetBits(board.pieceBitboards[WB]) >= 2) score += BISHOP_PAIR_BONUS;
    if (countSetBits(board.pieceBitboards[BB]) >= 2) score -= BISHOP_PAIR_BONUS;

    const int ROOK_OPEN_BONUS = 25;
    const int ROOK_SEMI_OPEN_BONUS = 10;
    Bitboard whitePawns = board.pieceBitboards[WP];
    Bitboard blackPawns = board.pieceBitboards[BP];

    Square sq;
    Bitboard wr = board.pieceBitboards[WR];
    while ((sq = pop_lsb(wr)) != NO_SQ) {
        int file = static_cast<int>(sq) % 8;
        Bitboard fileMask = FILE_A_BB << file;
        if (((whitePawns | blackPawns) & fileMask) == 0) score += ROOK_OPEN_BONUS;
        else if ((whitePawns & fileMask) == 0) score += ROOK_SEMI_OPEN_BONUS;
    }

    Bitboard br = board.pieceBitboards[BR];
    while ((sq = pop_lsb(br)) != NO_SQ) {
        int file = static_cast<int>(sq) % 8;
        Bitboard fileMask = FILE_A_BB << file;
        if (((whitePawns | blackPawns) & fileMask) == 0) score -= ROOK_OPEN_BONUS;
        else if ((blackPawns & fileMask) == 0) score -= ROOK_SEMI_OPEN_BONUS;
    }

    return score;
}


int quiescence_search(BoardState currentBoard, int alpha, int beta, int q_depth) {
    if (q_depth == 0) { 
        int staticEval = evaluate(currentBoard);
        return (currentBoard.sideToMove == WHITE) ? staticEval : -staticEval;
    }

    int stand_pat_score = evaluate(currentBoard);
    stand_pat_score = (currentBoard.sideToMove == WHITE) ? stand_pat_score : -stand_pat_score;

    if (stand_pat_score >= beta) {
        return beta; 
    }
    if (stand_pat_score > alpha) {
        alpha = stand_pat_score;
    }

    std::vector<Move> captureMoves;
    generate_legal_moves(currentBoard, captureMoves, true); 

    if (captureMoves.empty()) {
        return stand_pat_score; 
    }
    
    std::vector<ScoredMove> scoredCaptureMoves;
    for(const auto& cap_move : captureMoves) {
        int move_score = 1000;
        if (cap_move.pieceCaptured != NO_PIECE) {
            int victim = cap_move.pieceCaptured % NUM_BASE_PIECE_TYPES;
            int aggressor = cap_move.pieceMoved % NUM_BASE_PIECE_TYPES;
            move_score += mvv_lva[victim][aggressor];
        }
        scoredCaptureMoves.emplace_back(cap_move, move_score);
    }
    std::sort(scoredCaptureMoves.begin(), scoredCaptureMoves.end());


    for (const auto& scored_move : scoredCaptureMoves) {
        const Move& move = scored_move.move;
        BoardState nextBoard = currentBoard;
        make_move_internal(nextBoard, move);
        int score = -quiescence_search(nextBoard, -beta, -alpha, q_depth - 1);

        if (score > alpha) { 
            alpha = score;
        }
        if (alpha >= beta) {
            return beta; 
        }
    }
    return alpha; 
}


int negamax_search(BoardState currentBoard, int depth, int alpha, int beta, std::vector<Bitboard>& history) {
    Bitboard key = currentBoard.zobristKey;
    TTEntry& tt_entry = transpositionTable[key & (TT_SIZE - 1)];

    if (count_repetitions(history, key) >= 3) {
        return -CONTEMPT_FACTOR;
    }

    if (tt_entry.key == key && tt_entry.depth >= depth) {
        if (tt_entry.flag == TT_EXACT) {
            return tt_entry.score;
        }
        if (tt_entry.flag == TT_LOWER_BOUND && tt_entry.score >= beta) {
            return tt_entry.score;
        }
        if (tt_entry.flag == TT_UPPER_BOUND && tt_entry.score <= alpha) {
            return tt_entry.score;
        }
    }
    
    if (depth == 0) {
        return quiescence_search(currentBoard, alpha, beta, MAX_QUIESCENCE_PLY);
    }

    std::vector<Move> legalMoves;
    generate_legal_moves(currentBoard, legalMoves, false); 

    if (legalMoves.empty()) {
        if (is_king_in_check(currentBoard, currentBoard.sideToMove)) {
            return -(MATE_SCORE + depth); 
        } else {
            return 0; 
        }
    }
    
    std::vector<ScoredMove> scoredMoves;
    for(const auto& move : legalMoves) {
        int move_score = 0;
        if (move.isCapture) {
            move_score = 10000;
            if (move.pieceCaptured != NO_PIECE) {
                int victim = move.pieceCaptured % NUM_BASE_PIECE_TYPES;
                int aggressor = move.pieceMoved % NUM_BASE_PIECE_TYPES;
                move_score += mvv_lva[victim][aggressor];
            }
        } else if (move.isPromotion) {
            if (move.promotedTo == WQ || move.promotedTo == BQ) move_score = 9000;
            else move_score = 1000;
        }
        scoredMoves.emplace_back(move, move_score);
    }
    std::sort(scoredMoves.begin(), scoredMoves.end()); 

    int original_alpha = alpha;
    int maxScore = -INFINITE_SCORE; 

    for (const auto& scored_move : scoredMoves) {
        const Move& move = scored_move.move;
        BoardState nextBoard = currentBoard;
        make_move_internal(nextBoard, move);
        history.push_back(nextBoard.zobristKey);
        int score = -negamax_search(nextBoard, depth - 1, -beta, -alpha, history);
        history.pop_back();
        
        if (score > maxScore) {
            maxScore = score;
        }
        if (maxScore > alpha) {
            alpha = maxScore;
        }
        if (alpha >= beta) {
            break; 
        }
    }

    tt_entry.key = key;
    tt_entry.depth = depth;
    tt_entry.score = maxScore;
    if (maxScore <= original_alpha) {
        tt_entry.flag = TT_UPPER_BOUND;
    } else if (maxScore >= beta) {
        tt_entry.flag = TT_LOWER_BOUND;
    } else {
        tt_entry.flag = TT_EXACT;
    }

    return maxScore;
}

int search_root_move_task(BoardState boardAfterMove, int searchDepth, std::vector<Bitboard> history) {
    history.push_back(boardAfterMove.zobristKey);
    return -negamax_search(boardAfterMove, searchDepth - 1, -INFINITE_SCORE, INFINITE_SCORE, history);
}

Move find_best_move(const BoardState& board, int searchDepth, const std::vector<Bitboard>& history) {
    std::vector<Move> legalMoves;
    generate_legal_moves(board, legalMoves, false);

    if (legalMoves.empty()) {
        return Move(); 
    }

    std::vector<ScoredMove> rootScoredMoves;
     for(const auto& move : legalMoves) {
        int move_score = 0;
        if (move.isCapture) {
            move_score = 10000;
            if (move.pieceCaptured != NO_PIECE) {
                int victim = move.pieceCaptured % NUM_BASE_PIECE_TYPES;
                int aggressor = move.pieceMoved % NUM_BASE_PIECE_TYPES;
                move_score += mvv_lva[victim][aggressor];
            }
        } else if (move.isPromotion) {
            if (move.promotedTo == WQ || move.promotedTo == BQ) move_score = 9000;
            else move_score = 1000;
        }
        rootScoredMoves.emplace_back(move, move_score);
    }
    std::sort(rootScoredMoves.begin(), rootScoredMoves.end());

    Move bestMove = rootScoredMoves[0].move; 
    int bestScore = -INFINITE_SCORE; 

    std::cout << "Motor pensando com profundidade " << searchDepth << "..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<int>> futures;
    std::vector<Move> moves_in_order; 

    for (const auto& scored_move : rootScoredMoves) {
        const Move& move = scored_move.move;
        BoardState nextBoard = board;
        make_move_internal(nextBoard, move);
        auto historyCopy = history;
        futures.push_back(std::async(std::launch::async, search_root_move_task, nextBoard, searchDepth, historyCopy));
        moves_in_order.push_back(move);
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        try {
            int moveScore = futures[i].get(); 
            const Move& currentMove = moves_in_order[i];

            std::cout << "Lance Raiz: " << currentMove.toString() 
                      << " -> Pontuacao (perspectiva do jogador atual): " << moveScore << std::endl;

            if (moveScore > bestScore) {
                bestScore = moveScore;
                bestMove = currentMove;
            }
        } catch (const std::future_error& e) {
            std::cerr << "Erro no future: " << e.what() << " para o lance " << moves_in_order[i].toString() << std::endl;
        }
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    std::cout << "Tempo de busca: " << elapsed_time_ms / 1000.0 << "s" << std::endl;
    std::cout << "Motor (cor: " << (board.sideToMove == WHITE ? "Brancas" : "Pretas") 
              << ") escolheu: " << bestMove.toString() 
              << " com pontuacao (perspectiva do jogador atual): " << bestScore << std::endl;
    return bestMove;
}


int main() {
    initialize_zobrist_keys();
    initialize_attack_tables();
    initialize_evaluation_parameters();
    initialize_mvv_lva();
    clear_transposition_table();

    BoardState board;
    initialize_board(board);

    std::vector<Bitboard> positionHistory;
    positionHistory.push_back(board.zobristKey);

    std::string userInput;
    std::vector<Move> legal_moves_for_player;

    Color playerColor = WHITE;
    int currentSearchDepth = DEFAULT_SEARCH_DEPTH;

    while(true) {
        if (count_repetitions(positionHistory, board.zobristKey) >= 3) {
            std::cout << "EMPATE por repeticao de posicao!" << std::endl;
            break;
        }

        print_pretty_board(board);
        std::cout << "Avaliacao Estatica (Brancas): " << evaluate(board) << std::endl;
        std::cout << "Chave Zobrist: " << board.zobristKey << std::endl;

        if(board.sideToMove == BLACK && board.fullMoveNumber == 1) clear_transposition_table();

        generate_legal_moves(board, legal_moves_for_player, false); 

        if (legal_moves_for_player.empty()) {
            if (is_king_in_check(board, board.sideToMove)) {
                std::cout << "XEQUE-MATE! As " << (board.sideToMove == WHITE ? "Pretas" : "Brancas") << " venceram!" << std::endl;
            } else {
                std::cout << "AFOGAMENTO! Empate." << std::endl;
            }
            break; 
        }
        if (board.halfMoveClock >= 100) { 
            std::cout << "EMPATE pela regra dos 50 lances!" << std::endl;
            break;
        }
        
        Move chosenMove;
        if (board.sideToMove == playerColor) { 
            std::cout << "\nLances LEGAIS para as " << (board.sideToMove == WHITE ? "Brancas" : "Pretas") 
                      << " (" << legal_moves_for_player.size() << " lances):" << std::endl; 
            for (size_t i = 0; i < legal_moves_for_player.size(); ++i) { 
                std::cout << (i+1) << ". " << legal_moves_for_player[i].toString();
                 if (legal_moves_for_player[i].isCastleKingside) std::cout << " (O-O)";
                if (legal_moves_for_player[i].isCastleQueenside) std::cout << " (O-O-O)";
                if (legal_moves_for_player[i].isCapture && !legal_moves_for_player[i].isEnPassant) { 
                    std::cout << " (x" << piece_to_char(legal_moves_for_player[i].pieceCaptured) << ")";
                }
                if (legal_moves_for_player[i].isEnPassant) {
                    std::cout << " (ep x" << piece_to_char(legal_moves_for_player[i].pieceCaptured) << ")";
                }
                std::cout << "  ";
                if ((i + 1) % 5 == 0) std::cout << std::endl; 
            }
            std::cout << std::endl;

            std::cout << "Seu lance (ex: e2e4) ou 'quit': ";
            std::cin >> userInput;

            if (userInput == "quit") break;
            chosenMove = parse_move_input(userInput, legal_moves_for_player, board.sideToMove);
            if (chosenMove.fromSquare == NO_SQ) {
                 std::cout << "Lance invalido. Tente novamente." << std::endl;
                 std::cin.clear(); 
                 std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
                 continue; 
            }
        } else { 
            chosenMove = find_best_move(board, currentSearchDepth, positionHistory);
            if (chosenMove.fromSquare == NO_SQ) {
                std::cout << "Motor nao conseguiu encontrar um lance!" << std::endl;
                break;
            }
        }

        make_move_internal(board, chosenMove);
        positionHistory.push_back(board.zobristKey);
    }
    
    return 0;
}
