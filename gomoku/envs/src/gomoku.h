#pragma once

#include <tuple>
#include <vector>
#include <new>

class Gomoku {
public:
  using move_type = int;
  using board_type = std::vector<std::vector<int>>;

  Gomoku(unsigned int n, unsigned int n_in_row, int first_color);

  bool has_legal_moves();
  std::vector<int> get_legal_moves();
  void execute_move(move_type move);
  std::vector<int> get_game_status();
  void display() const;
  void restore(board_type reboard, int color);

  inline unsigned int get_action_size() const { return this->n * this->n; }
  inline board_type get_board() const { return this->board; }
  inline move_type get_last_move() const { return this->last_move; }
  inline int get_current_color() const { return this->cur_color; }
  inline unsigned int get_n() const { return this->n; }
  
  void set_board_data(float* board){
    int size = this->n;
    for (int i = 0; i < size; i++)  
    {
      for (int j = 0; j < size; j++)
      {
        board[i*size+j] = this->board[i][j];
      }
    }
  }

private:
  board_type board;      // game borad
  unsigned int n;        // board size
  unsigned int n_in_row; // 5 in row or else

  int cur_color;       // current player's color
  move_type last_move; // last move
};