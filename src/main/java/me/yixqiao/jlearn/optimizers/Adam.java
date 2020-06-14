package me.yixqiao.jlearn.optimizers;

import me.yixqiao.jlearn.matrix.Matrix;

public class Adam extends Optimizer {
    double learningRate;
    double beta1 = 0.9, beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;
    Matrix velocity;

    public Adam(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public Optimizer cloneOptimizer() {
        return new Adam(learningRate);
    }

    @Override
    public void multiplyLR(double d) {
        this.learningRate *= d;
    }

    private void init(Matrix g) {
        velocity = new Matrix(g.rows, g.cols);
    }

    @Override
    public Matrix apply(Matrix g) {
        if (t==0)
            init(g);
        t++;
        



        return velocity;
    }

    // while (1):					#till it gets converged
    // 	t+=1
    // 	g_t = grad_func(theta_0)		#computes the gradient of the stochastic function
    // 	m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
    // 	v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
    // 	m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
    // 	v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates
    // 	theta_0_prev = theta_0
    // 	theta_0 = theta_0 - (alpha*m_cap)/(math.sqrt(v_cap)+epsilon)	#updates the parameters
    // 	if(theta_0 == theta_0_prev):		#checks if it is converged or not
    // 		break

}
