#include <iostream>
#include <casadi/casadi.hpp>

using namespace casadi;
using namespace std;


#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;




MatrixXd Calculate_inducing_points(const double scale, const Eigen::MatrixXd &x_) {
    MatrixXd U(int(x_.rows() / scale), x_.cols()); // Initialize U

    double x_max;
    double x_min;
    for (int i = 0; i < x_.cols(); ++i) {
        x_max = x_.col(i).maxCoeff();

        x_min = x_.col(i).minCoeff();
        U.col(i) = Eigen::VectorXd::LinSpaced(x_.size() / scale, x_min, x_max);

    }

    return U;
}

MatrixXd SquaredExponential(const VectorXd &theta, const Eigen::MatrixXd &x1,
                                                    const Eigen::MatrixXd &x2) {

    MatrixXd dist = MatrixXd::Zero(x1.rows(), x2.rows());
    
    
    for (int d = 0; d < x1.cols(); ++d) {
        for (int i = 0; i < x1.rows(); ++i) {
            for (int j = 0; j < x2.rows(); ++j) {
                dist(i, j) += (x1(i, d) - x2(j, d)) *
                              (x1(i, d) - x2(j, d)) * (1 / pow(theta[d + 1], 2));


            }
        }
    }

    MatrixXd K = pow(theta[0], 2) * (-0.5 *
                                     dist).array().exp();    // K_x1x2 = σ^2 * exp(-0.5|x1-x2|/l^2)

    //add noise for numerical issues with log-likehood calculation
    for (int i = 0; i < x1.rows(); ++i) {
        for (int j = 0; j < x2.rows(); ++j) {
            if (i == j)
                K(i, j) += 0.001;
        }
    }


    return K;
}







double SineWave(double x, double f, double mean, double noise)   //Modified by Hakim
    {
  




        double signal = sin(f * x);
       // double noisy_signal = signal + distribution(generator);

        return signal;
    }


Eigen::MatrixXd predict_test(const Eigen::MatrixXd x_s, 
	           const Eigen::MatrixXd x_, 
	           const Eigen::MatrixXd y_,
	           const Eigen::MatrixXd u_,
	           const Eigen::VectorXd theta_,
	           double lamb) {

    int n = y_.rows();

    double sigma_ = 0.0001;
    //double lamb= 1.0;

    //Eigen::MatrixXd Lambda_m = Eigen::MatrixXd::Identity(y_.rows(), y_.rows());

     MX Lambda_m = opti.variable(y_.rows(), y_.rows());

    MatrixXd K_xs = SquaredExponential(theta_, x_, x_s);
    MatrixXd K_us = SquaredExponential(theta_, u_, x_s);
    MatrixXd K_su = SquaredExponential(theta_, x_s, u_);

    MatrixXd K_xu = SquaredExponential(theta_, x_, u_);
    MatrixXd K_ux = SquaredExponential(theta_, u_, x_);

    MatrixXd K_uu = SquaredExponential(theta_, u_, u_);
    MatrixXd K_xx = SquaredExponential(theta_, x_, x_);
    MatrixXd K_ss = SquaredExponential(theta_, x_s, x_s);

    MatrixXd I = Eigen::MatrixXd::Identity(u_.rows(), u_.rows());
    MatrixXd iK_uu = Eigen::MatrixXd::Identity(u_.rows(), u_.rows());
    MatrixXd L_B = Eigen::MatrixXd::Identity(u_.rows(), u_.rows());
    MatrixXd B_lambda = Eigen::MatrixXd::Identity(u_.rows(), u_.rows());
    MatrixXd iB_lambda = K_uu + pow(sigma_, -2) * (K_xu.transpose() * Lambda_m * K_xu);
    
    for (int i = n - 2; i >= 0; i--) {
        Lambda_m(i, i) = Lambda_m(i + 1, i + 1) * lamb;
    }
   

    for (size_t i = 0; i < iB_lambda.rows(); ++i) {
        for (size_t j = 0; j <= i; ++j) {
            //L_B(i, j) = iB_lambda(i,j);
            L_B(i, j) = K_uu(i, j);
        }
    }
    // compute inverses using Chelosky decomp
    B_lambda = iB_lambda.llt().solve(I);
    iK_uu = K_uu.llt().solve(I);

   

    MatrixXd mu = pow(sigma_, -2) * K_us.transpose() * B_lambda * K_xu.transpose() * Lambda_m * y_;

    return {mu};
}

int main(){


    casadi::DM lambda;
   

    int num_samples = 10;


    Eigen::VectorXd X_data = Eigen::VectorXd::LinSpaced(num_samples, 0, 2 * M_PI);

    Eigen::VectorXd Y_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        Y_data[i] = SineWave(X_data[i], 1.0, 0.0, 0.0);
    }


   Eigen::VectorXd theta_0(2);
    theta_0 << 1.0 , 1.0; 

    double scale = 0.5;


   
    Eigen::VectorXd U = Calculate_inducing_points(scale,X_data);

    Eigen::VectorXd X_s(1);
    X_s << 2;    

    casadi::DM mu = predict(X_s, X_data , Y_data , U, theta_0, lambda) ;
    

    double Y_g = SineWave(X_s(0), 1.0, 0.0, 0.0);
    std::cout << "prediction GP: "<< mu <<std::endl;
    std::cout << "error GP: "<<  abs(mu - Y_g) <<std::endl;


  /*


	Opti opti = casadi::Opti(); // Optimization problem
	  
	Slice all;
	  // ---- decision variables --------- since its multiple shooting x and u are both decision variables
	MX lambda = opti.variable(1, 1); // control trajectory (2 thrusters)
	  

	MX J = 0;    //Initialize cost function
     



    J = (mu-Feat)*(mu-Feat);




	  
	
	  opti.minimize(J); // add J to OCP

	 // opti.subject_to(-10<=lambda<=10);           // control is limited  to 1 N 


    Dict opts;

    // Set the options in the dictionary
    opts["ipopt.print_level"] = 0;
    opts["print_time"] = 0;
    opts["ipopt.sb"] = "yes";

    // Set solver options using the dictionary
    opti.solver("ipopt", opts);
	   //opti.solver("ipopt"); // set numerical backend



	 OptiSol sol = opti.solve();   // actual solve


	  auto solution =  std::make_unique<casadi::OptiSol>(opti.solve());

	      
*/

       

  return 0;
}