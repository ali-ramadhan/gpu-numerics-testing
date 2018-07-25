#include <iostream>
#include <fstream>
#include <random>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

float normal_pdf(float x, float m, float s) {
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;

    return (inv_sqrt_2pi / s) * std::exp(-0.5f * a * a);
}

void solve_1D_wave_equation(double mu, double sigma, string output_filepath) {
    IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "");
    ofstream outfile(output_filepath);

    /* Problem parameters */
    double c = 1.0;  // Propagation speed of the wave.
    double L = 1.0;  // Length of the domain.
    int N = 100;     // Number of grid points.

    double dx = L/N;
    double dt = 0.01;
    double t_end = 1.0;

    double alpha = c*c * dt*dt / (2*dx*dx);

    /* We'll discretize the wave equation using the Crank-Nicolson method and
     * solve the resulting linear system A*u = b at every time step.
     */
    MatrixXf A(N,N);
    VectorXf b(N);
    VectorXf u_n(N);
    VectorXf u_nm1(N);

    // Initialize matrix of coefficients.
    A(0,0)     = 1 + 2*alpha;
    A(N-1,N-1) = 1 + 2*alpha;
    A(0,1)     = -alpha;
    A(N-1,N-2) = -alpha;

    for (int i = 1; i < N-1; i++) {
        A(i,i) = 1 + 2*alpha;
        A(i,i+1) = -alpha;
        A(i,i-1) = -alpha;
    }

    // Set initial conditions.
    u_nm1(0)   = 0;
    u_nm1(N-1) = 0;
    for (int i = 1; i < N-1; i++)
        u_nm1(i) = normal_pdf(i*dx, mu, sigma);

    // Output first row corresponding to initial condition (t = 0)
    outfile << u_nm1.format(CommaInitFmt) << '\n';

    // Take the first time step (requires special scheme).
    u_n(0)   = 0;
    u_n(N-1) = 0;

    for (int i = 1; i < N-1; i++)
        u_n(i) = u_nm1(i) + (c*c/2) * (u_nm1(i+1) - 2*u_nm1(i) + u_nm1(i-1));

    // Output second row corresponding to first time step.
    outfile << u_n.format(CommaInitFmt) << '\n';

    double t = dt; // We already took one step so t = dt now.

    while (t < t_end) {
        t += dt;

        // Set up right-hand side vector b.
        b(0)   = 2*(1-alpha)*u_n(0) - u_nm1(0) + alpha*u_n(1);
        b(N-1) = 2*(1-alpha)*u_n(N-1) - u_nm1(N-1) + alpha*u_n(N-1);
        for (int i = 1; i < N-1; i++)
            b(i) = 2*(1-alpha)*u_n(i) - u_nm1(i) + alpha*(u_n(i+1) + u_n(i-1));

        u_nm1 = u_n;
        
        VectorXf u_np1 = A.colPivHouseholderQr().solve(b);
        
        u_np1(0)   = 0;
        u_np1(N-1) = 0;

        outfile << u_np1.format(CommaInitFmt) << '\n';
        
        u_n = u_np1;
    }
    
    outfile.close();
}

void solve_1D_wave_equations(int M, float *mu, float *sigma) {
    # pragma acc kernels
    {
    for (int i = 0; i < M; i++) {
        string file_suffix = to_string(i);
        file_suffix.insert(file_suffix.begin(), 3 - file_suffix.length(), '0');
        string filename = "cpu_wave_" + file_suffix + ".dat";

        cout << "Solving wave equation problem " << i << "..." << " (mu=" << mu[i] << ", sigma="
             << sigma[i] << ")\n";

        solve_1D_wave_equation(mu[i], sigma[i], filename);
    }
    }
}

int main() {
    int M = 25;  // Number of 1D wave equation problems to solve.

    std::random_device rd;  // Obtain a random number generator from hardware.
    std::mt19937 mt(rd()); // Seed the Mersenne Twister generator.

    /* We will impose a Gaussian wave initial condition for the 1D wave equation, with randomly
     * generated means (0.2 < mu < 0.8) and standard deviations (0.01 < sigma < 0.5).
     */
    std::uniform_real_distribution<float> uniform_mu(0.2, 0.8);
    std::uniform_real_distribution<float> uniform_sigma(0.01, 0.5);

    float *mu = new float[M];
    float *sigma = new float[M];

    for(int i = 0; i < M; i++) {
        mu[i] = uniform_mu(mt);
        sigma[i] = uniform_sigma(mt);
    }

    // Solve the M problems one-by-one on the CPU.
    solve_1D_wave_equations(M, mu, sigma);

    delete [] mu;
    delete [] sigma;
}
