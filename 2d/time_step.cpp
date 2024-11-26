# include "iteration.cpp"

class time_step{
    private:
        double time;
        int time_count;

    public:
        double get_time(){
            // gets the current time in seconds
            return( time );
        }

        int get_time_step(){
            // returns what time step we are on
            return( time_count );
        }

        void run_time_step(){
            // runs a single time step and calls itteration iteration.cpp
            double x = 0;
        }

        void run(){
            // runs the problem for a collection of time steps
            double t = 0;
        }
};