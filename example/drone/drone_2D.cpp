#include <iostream>
#include <casadi/casadi.hpp>

using namespace casadi;
using namespace std;


//TODO
//1-Add reference matrix Done
//2-Add Weight matrix  done
//3-Add simulation loop
//4-correct naming
//5-make good design of software OOP
//6-include direct shooting
//7-create ROV example
//8-create ros node

//system params

const double m = 1;   //drone mass
const double J = 1;   //drone angular inertia
const double l = 1;   //drone length



//OCP params
const int N_s = 6;   //number of states
const int N_c = 2;   //number of controls
const double T = 0.5;   //Prediction Horizon 
const int N = 20; // number of control steps
const int Ns = 100; // number of simulation steps



//

//define state transition equation: dx/dt = f(x,u)
//6 states
//x1 = x
//x2 = x_dot
//x3 = y
//x4 = y_dot
//x5 = theta
//x6 = theta_dot

//SS_equation
//x1_d = x2
//x2_d = ((u1+u2) * cos(x5))/m
//x3_d = x4
//x4_d = ((u1+u2) * sin(x5))/m
//x5_d = x6
//x6_d = ((u2 * l - u1 * l)/J

//       |                         |
///     /|\ u1                    /|\ u2 
//       |          _____          |                   
//      |_|--------|__m__|--------|_|


MX f(const MX& x, const MX& u) {
  //return vertcat(x(1), u-x(1));
   return vertcat(x(1), 
                  ((u(0)+u(1)) * sin(x(4)))/m , 
                   x(3),
                   ((u(0)+u(1)) * cos(x(4)))/m,
                   x(5),
                   (u(1) * l - u(0) * l) / J
                   );
}


std::vector<double> f_sim(const std::vector<double>& x, const std::vector<double>& u) {
    std::vector<double> f(6); // Create a vector with 6 elements

    f[0] = x[1];
    f[1] = ((u[0] + u[1]) * sin(x[4])) / m;
    f[2] = x[3];
    f[3] = ((u[0] + u[1]) * cos(x[4])) / m;
    f[4] = x[5];
    f[5] = (u[1]  - u[0] ) ;

    return f;
}


   double x_sim = 0.0;
   double vx_sim = 0.0;
   double y_sim = 0.0;
   double vy_sim = 0.0;
   double th_sim = 0.0;
   double w_sim = 0.0;

   
int main(){



    // ---- initial conditions --------// start at position 0,0 with 0 velocity ...
   MX X_i = MX::zeros(N_s,1);
  // Drone trajectory following
  // ----------------------
  // An optimal control problem (OCP),
  // solved with direct multiple-shooting, TODO: direct single shooting
  //
   

   //define reference trajectory
   
   MX X_r = MX::zeros(N_s, 1);

   X_r(0,0) = 1.0;
   X_r(1,0) = 0.0;
   X_r(2,0) = 1.0;
   X_r(3,0) = 0.0;
   X_r(4,0) = 0.0;
   X_r(5,0) = 0.0;


   //define MPC weights

   MX Q = MX::zeros(N_s, N_s);

   Q(0,0) = 100;
   Q(1,1) = 1.0;
   Q(2,2) = 100;
   Q(3,3) = 1.0;
   Q(4,4) = 1.0;
   Q(5,5) = 1.0;


   MX X_rs = X_r;

  for (int i = 0; i < N ; ++i)
  {
    X_rs = horzcat(X_rs, X_r);

  }

  MX XRR = X_rs;



  // Simulation step for receding horizon controller
   // Simulate the system for N_s steps, after each step compute the optimal control trajectory u*_traj
   // use the first action in this optimal sequence (U_star) to simulate your system, the simulated state at the next time step 
   // is then used the initial condition at the next optimization performed. 



  for (int k_sim = 0; k_sim < Ns; ++k_sim) {
	  Opti opti = casadi::Opti(); // Optimization problem
	  
	  Slice all;
	  // ---- decision variables --------- since its multiple shooting x and u are both decision variables
	  MX X = opti.variable(N_s, N + 1); // state trajectory
	  MX U = opti.variable(N_c, N); // control trajectory (2 thrusters)
	  
	  // ---- objective

	  MX J = 0;    //Initialize cost function

	  // calculate cost J using least-square error  J = (x-x_ref)T.Q.(x-x_ref)
	 for (int k = 0; k < N; ++k) {

	    MX  a = mtimes(transpose(X(all,k)-X_rs(all,k)),Q);
	    J = J + mtimes(a,X(all,k)-X_rs(all,k));
	 }

	  
	  
	  
	  opti.minimize(J); // add J to OCP
	  double dt = T / N; //step size

	  // ---- dynamic constraints -------- RK4 integration scheme
	  
	  for (int k = 0; k < N; ++k) {
	    MX k1 = f(X(all,k),U(all,k));
	    MX k2 = f(X(all,k)+dt/2*k1, U(all,k));
	    MX k3 = f(X(all,k)+dt/2*k2, U(all,k));
	    MX k4 = f(X(all,k)+dt*k3,   U(all,k));
	    MX x_next = X(all,k) + dt/6*(k1+2*k2+2*k3+k4);
	    opti.subject_to(X(all,k+1)==x_next); // close the gaps 
	  }

	  // ---- thrusters limit constraints -----------
	  opti.subject_to(-10<=U<=10);           // control is limited  to 1 N 


    Dict opts;

    // Set the options in the dictionary
    opts["ipopt.print_level"] = 0;
    opts["print_time"] = 0;
    opts["ipopt.sb"] = "yes";

    // Set solver options using the dictionary
    opti.solver("ipopt", opts);
	   //opti.solver("ipopt"); // set numerical backend


	 std::vector<double> state_sim(6, 0.0); // Creates a vector with 6 elements, all set to 0.0  
	  
	        
	 opti.subject_to(X(all,0)==X_i);  // Set inital condition
 
	 opti.subject_to(X(2,all)>0.0);   // Dont hit the ground (y>0)



	 
	  // ---- set initial guess for solver---
	  

	  // ---- solve NLP              ------
	     OptiSol sol = opti.solve();   // actual solve


	     auto solution =  std::make_unique<casadi::OptiSol>(opti.solve());
	     auto U_star1 = solution->value(U)(0, 0);
	     auto U_star2 = solution->value(U)(1, 0);

	     cout << "U_star " <<solution->value(U)(all, 0) <<endl;
	     auto U_sall = solution->value(U)(all, 0);

	     double u_s1  = static_cast<double>(U_star1);
	     double u_s2 = static_cast<double>(U_star2);
	      

	      std::vector<double> u_sim {u_s1 ,u_s2};

	      MX U_star = MX::zeros(2, 1);

	      U_star(0,0) = u_s1;
	      U_star(1,0) = u_s2;
	      


          
          state_sim[0]= x_sim;
          state_sim[1]= vx_sim;
          state_sim[2]= y_sim;
          state_sim[3]= vy_sim;
          state_sim[4]= th_sim;
          state_sim[5]= w_sim;

          x_sim = x_sim + dt * f_sim(state_sim, u_sim)[0];
          vx_sim = vx_sim + dt * f_sim(state_sim, u_sim)[1]; 



          y_sim = y_sim +  dt*f_sim(state_sim, u_sim)[2];
         
          vy_sim = vy_sim + dt * f_sim(state_sim, u_sim)[3];
          th_sim = th_sim + dt * f_sim(state_sim, u_sim)[4];
          w_sim = w_sim + dt * f_sim(state_sim, u_sim)[5];


      
	     //get feedback 
	  
	      X_i(0,0) = x_sim; 
	      X_i(1,0) = vx_sim; 
	      X_i(2,0) = y_sim; 
	      X_i(3,0) = vy_sim; 
	      X_i(4,0) = th_sim; 
	      X_i(5,0) = w_sim; 

         //cout << "y pos : " <<y_sim<<endl;
         //cout << "x pos : " <<x_sim<<endl;
         //cout << "theta : " <<th_sim<<endl;
         //cout << "omega : " <<w_sim<<endl;
         //cout << "torque : " <<f_sim(state_sim, u_sim)[5]<<endl;
         //cout << "u : " <<u_sim;
         



	     }

	     cout << "x pos final: " <<x_sim <<endl;
	     cout << "vx  final: " <<vx_sim<<endl;
	     cout << "y pos final: " <<y_sim<<endl;
	     cout << "vy final: " <<vy_sim<<endl;
         cout << "theta final: " <<th_sim<<endl;
	     cout << "angular rate final " <<w_sim<<endl;



  return 0;
}