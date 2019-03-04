OPENCV = `pkg-config --libs --cflags opencv`
% : %.cpp
	g++ -std=c++11 $< $(OPENCV) -o $@ 
clean:
	rm -rf *.o 
	rm -rf *~ *# .#*
	rm -rf main main_test_case_1 main_test_case_2 main_test_case_3 error_detection
