Note.Before run the any program please verify the file path.

For creating State for Tic-Tac-Toe Game ,follow the below steps.

Step:1 Run the "GenerateAllPossibleGame.java" by updating file path.It will give following output and file with moves.
	Total No of Different Game 
	Total win in 5th move : 1440
	Total win in 6th move : 5328
	Total win in 7th move : 47952
	Total win in 8th move : 72576
	Total win in 9th move : 81792
	Draw Games : 46080
	
Step2:After genreating all move.Merge Even move and odd move in two separate files.DrawGames add in oddMove(OddMove.text & EvenMove.text).

Step3:Run the "RewardGameState.java" with OddMove.txt and EvenMove.txt .It will generate the "AllMoveWithReward.txt".

Step4:Run the "RemoveDuplicateState.java" with "AllMoveWithReward.txt".It will remove the duplicate state and update probability in "DuplicateRemoved.txt".

Step5:Run the "TicTacToeGameTrainer.java" with "DuplicateRemoved.txt" .It will update the probability table for Both position(i.e AI play as First Player and Second Player).It will generate "SmartAIMove.csv" .

1)First time run with "DuplicateRemoved.txt" and Result will be.
	   Total Game :9009
       X Player:602
       O Player:220
       XXDrawOO:8187	   
2)Second time run with "SmartAIMove.csv" and Result will be.  
	   Total Game :9009
       X Player:10
       O Player:4
       XXDrawOO:8995
3)Third time run with "SmartAIMove.csv" and Result will be.  
	   Total Game :9009
       X Player:0
       O Player:0
       XXDrawOO:9009
	   
Step6:Now Run "TicTacToGame.java"  with "SmartAIMove.csv".
