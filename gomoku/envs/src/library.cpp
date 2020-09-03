#include <gomoku.h>
#include <iostream>

extern "C"  //Tells the compile to use C-linkage for the next scope.
{
  Gomoku* G_new(int n, int n_in_row, int first_color){return new Gomoku(n, n_in_row, first_color);}
  
  int G_move(Gomoku &gomoku, int action){
    gomoku.execute_move(action);
    return 1;
  }

  int G_board(Gomoku &gomoku, float* pboard){
    gomoku.set_board_data(pboard);
    return 1;
  }

  int G_status(Gomoku &gomoku, float* pstatus){
    // test display
    std::vector<int> status = gomoku.get_game_status();
    pstatus[0] = status[0];
    pstatus[1] = status[1];
    return 1;
  }

  int G_display(Gomoku &gomoku){

    // test display
    gomoku.display();
    return 1;
  }

  int G_clean(Gomoku &gomoku){
    delete &gomoku;
    return 1;
  }
    
  

} //End C linkage scope.