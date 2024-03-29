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


double CosWave(double x, double f, double mean, double noise)   //Modified by Hakim
    {
  

        double signal = 0.5 * cos(f * x);
       // double noisy_signal = signal + distribution(generator);

        return signal;
    }

Eigen::MatrixXd predict_test(const Eigen::MatrixXd x_s, 
	           const Eigen::MatrixXd x_, 
	           const Eigen::MatrixXd y_,
	           const Eigen::MatrixXd u_,
	           const Eigen::VectorXd theta_,
	           double lamb,
	           double sigma_ ) {

    int n = y_.rows();

    //double lamb= 1.0;

    //Eigen::MatrixXd Lambda_m = Eigen::MatrixXd::Identity(y_.rows(), y_.rows());

    
    MatrixXd Lambda_m = Eigen::MatrixXd::Identity(y_.rows(), y_.rows());

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
   
    //std::cout <<Lambda_m <<std::endl;
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



casadi::MX predict(const Eigen::MatrixXd x_s, 
	           const Eigen::MatrixXd x_, 
	           const Eigen::MatrixXd y_,
	           const Eigen::MatrixXd u_,
	           const Eigen::VectorXd theta_,
	           casadi::MX lamb,
	           Opti opti ,
	           double sigma_ ) {

    int n = y_.rows();
    
    //double lamb= 1.0;
   
     MX Lambda_m_ca(x_.rows(), x_.rows());

     



    MX lambda_ca = opti.variable(1, 1);


    MatrixXd K_xs = SquaredExponential(theta_, x_, x_s);
    MatrixXd K_us = SquaredExponential(theta_, u_, x_s);
    MatrixXd K_su = SquaredExponential(theta_, x_s, u_);

    MatrixXd K_xu = SquaredExponential(theta_, x_, u_);
    MatrixXd K_ux = SquaredExponential(theta_, u_, x_);

    MatrixXd K_uu = SquaredExponential(theta_, u_, u_);
    MatrixXd K_xx = SquaredExponential(theta_, x_, x_);
    MatrixXd K_ss = SquaredExponential(theta_, x_s, x_s);

    MatrixXd Lambda_m = Eigen::MatrixXd::Identity(x_.rows(), x_.rows());

    



    for (int i = n - 2; i >= 0; i--) {
        Lambda_m_ca(i, i) = Lambda_m(i + 1, i + 1) * lamb;
    }
     



    
    
    MX  K_uu_ca(u_.rows(), u_.rows());

    MX  K_us_ca(u_.rows(), x_s.rows());
    MX K_xu_ca(x_.rows(), u_.rows());
    MX K_su_ca(x_s.rows(), u_.rows());
    MX K_ux_ca(u_.rows(), x_.rows());

    //MX B_lambda_ca(x_.rows(), x_.rows());

    MX y_ca(y_.rows(), 1);
    
    for (int i = 0; i < u_.rows(); ++i) {
        for (int j = 0; j < x_s.rows(); ++j) {
            K_us_ca(i, j) = K_us(i, j);
        }
    }


    for (int i = 0; i < x_.rows(); ++i) {
    for (int j = 0; j < u_.rows(); ++j) {
            K_xu_ca(i, j) = K_xu(i, j);
        }
    }

    for (int i = 0; i < u_.rows(); ++i) {
    for (int j = 0; j < x_.rows(); ++j) {
            K_ux_ca(i, j) = K_ux(i, j);
        }
    }


    for (int i = 0; i < u_.rows(); ++i) {
    for (int j = 0; j < u_.rows(); ++j) {
            K_uu_ca(i, j) = K_uu(i, j);
        }
    }

    for (int i = 0; i < x_s.rows(); ++i) {
    for (int j = 0; j < u_.rows(); ++j) {
            K_su_ca(i, j) = K_su(i, j);
        }
    }



    for (int j = 0; j < y_.rows(); ++j) {
        y_ca(j) = y_(j);
    }

    
   // MX iB_lambda_m = K_uu_ca + pow(sigma_, -2) * (K_ux_ca * Lambda_m_ca * K_xu_ca);
  

     MX _A = pow(sigma_, -2) * mtimes(K_ux_ca , Lambda_m_ca);
     MX _B = mtimes(_A, K_xu_ca);
     MX iB_lambda_m  = K_uu_ca + _B; 

     //MX B_lambda_ca = Inverse (iB_lambda_m);

    MX B_lambda_ca =solve(iB_lambda_m, MX::eye(iB_lambda_m.size1()));



    
    MX _C = pow(sigma_, -2) * mtimes(K_su_ca , B_lambda_ca);  //1xm   mxm
    MX _D = mtimes(K_ux_ca , Lambda_m_ca); //mxn   nxn  (m,n)
    
    MX _E = mtimes(_C , _D); //1xm   mxn
    MX mu = mtimes(_E , y_ca); //mxn  nx1
    
  

   

    //  mu = pow(sigma_, -2) * K_us.transpose() * B_lambda * K_xu.transpose() * Lambda_m * y_;
    return {mu};
}

int main(){

    Opti opti = casadi::Opti(); 
    casadi::MX lambda = opti.variable(1, 1);
   

    int n = 30;
    int n_s = 1;
    double W_lambda =1.0;

    Eigen::VectorXd X_data = Eigen::VectorXd::LinSpaced(n, 0, 2 * M_PI);

    Eigen::VectorXd Y_data(n);
    for (int i = 0; i < n; ++i) {
        Y_data[i] = SineWave(X_data[i], 1.0, 0.0, 0.0);

        if (i>n/2)
        {
        	Y_data[i] = CosWave(X_data[i], 1.0, 0.0, 0.0);
        }
    }
    



  
   Eigen::VectorXd theta_0(2);
    theta_0 << 1.0 , 1.0; 

    double scale = 2.0;
    double sigma_ = 0.001;

    Eigen::VectorXd U = Calculate_inducing_points(scale,X_data);
    
    Eigen::VectorXd X_s(1);

    X_s << X_data[n-1]; 


    casadi::MX mu ;
    
	MX J = 0;
    
    for ( int i=0; i< n_s; i++)
         
         {      

         	    X_s << X_data[n-1]; 
         	    double Y_g = SineWave(X_s(0), 1.0, 0.0, 0.0) + CosWave(X_s(0), 1.0, 0.0, 0.0);
               
         	    casadi::MX mu =  predict(X_s, X_data , Y_data , U, theta_0, lambda, opti,sigma_) ;



	           J = ((W_lambda)*(mu-Y_g)*(mu-Y_g));


		  }




    






    //std::cout << "prediction GP: "<< mu <<std::endl;
    //std::cout << "error GP: "<<  abs(mu - Y_g) <<std::endl;

    




    opti.subject_to(0.5<=lambda<=1.0);
    opti.minimize(J);

    Dict opts;


    // Set the options in the dictionary
    opts["ipopt.print_level"] = 5;
    opts["print_time"] = 1;
    opts["ipopt.sb"] = "yes";
    opts["ipopt.max_iter"] = 100; // Set the maximum number of iterations (adjust as needed)
    // Set solver options using the dictionary
    opti.solver("ipopt", opts);
 
    //OptiSol sol = opti.solve();   // actual solve

	auto solution =  std::make_unique<casadi::OptiSol>(opti.solve());
    cout << "lambda_opt: " <<solution->value(lambda)(0, 0) <<endl;
    


        //Test1
    double cost1=0.0;
    double cost2=0.0;
    double cost3=0.0;
    Eigen:MatrixXd mu_test;
    for ( int i=0; i< 1; i++)
         
         {      
                double l=0.7;
         	    X_s << X_data[n-1] ; 
         	    double Y_g = SineWave(X_s(0), 1.0, 0.0, 0.0)+ CosWave(X_s(0), 1.0, 0.0, 0.0);

         	    mu_test =  predict_test(X_s, X_data , Y_data , U, theta_0, l,sigma_) ;

	            

	           cost1 = (mu_test(0,0)-Y_g)*(mu_test(0,0)-Y_g);


		  }

    for ( int i=0; i< n_s; i++)
         
         {      
                double l=0.8;
         	    X_s << X_data[n-1]; 
         	    double Y_g = SineWave(X_s(0), 1.0, 0.0, 0.0)+CosWave(X_s(0), 1.0, 0.0, 0.0);

         	    mu_test =  predict_test(X_s, X_data , Y_data , U, theta_0, l,sigma_) ;

                
	           cost2 = (mu_test(0,0)-Y_g)*(mu_test(0,0)-Y_g);


		  }


		      for ( int i=0; i< n_s; i++)
         
         {      
                double l=1.0;
         	    X_s << X_data[n-1]; 
         	    double Y_g = SineWave(X_s(0), 1.0, 0.0, 0.0)+CosWave(X_s(0), 1.0, 0.0, 0.0);

         	    mu_test =  predict_test(X_s, X_data , Y_data , U, theta_0, l,sigma_) ;

	

	           cost3 = (mu_test(0,0)-Y_g)*(mu_test(0,0)-Y_g);


		  }
         double cost4 = 0.0;
		      for ( int i=0; i< n_s; i++)
         
         {      
               // double l=double(0.1);
               double l=double(solution->value(lambda)(0, 0));
         	   //double l=1.0;
         	    X_s << X_data[n-1]; 
         	    double Y_g = SineWave(X_s(0), 1.0, 0.0, 0.0)+CosWave(X_s(0), 1.0, 0.0, 0.0);

         	    mu_test =  predict_test(X_s, X_data , Y_data , U, theta_0, l,sigma_) ;

	

	           cost4 =  (mu_test(0,0)-Y_g)*(mu_test(0,0)-Y_g);


		  }

      
         cout << "Cost lambda optimal: " << solution->value(J)(0, 0) << endl;
         cout << "Cost 0.7: " << cost1 << endl;
         cout << "Cost lambda = 0.8: " << cost2 << endl;
         cout << "Cost lambda = 1.0: " << cost3 << endl;
         cout << "Cost lambda after substitution = : " << cost4 <<endl;

  return 0;
}