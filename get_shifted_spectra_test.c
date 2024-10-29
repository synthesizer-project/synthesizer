#include <stdio.h>
#include <stdlib.h>
#include <math.h>  // Include math.h for abs()

#define N_LAM 5  // Number of wavelength bins
#define N_PARTICLES 3  // Number of particles

typedef struct {
    double *wavelengths;  // Array of wavelengths
    int nlam;  // Number of wavelength bins
} Grid;

typedef struct {
    double *masses;  // Array of particle masses
    double *fesc;  // Array of escape fractions
    int npart;  // Number of particles
} Particles;

// Function prototype for find_nearest_wavelength_bin
int find_nearest_wavelength_bin(double lambda, double *grid_wavelengths, int nlam);

void spectra_loop_cic_serial(Grid *grid, Particles *parts, double *spectra, double *part_velocities) {
    int nlam = grid->nlam;  // Number of wavelengths
    int npart = parts->npart;  // Number of particles
    double c = 3e8;  // Speed of light in m/s

    // Get grid wavelengths (assuming rest-frame wavelengths are stored in grid)
    double *grid_wavelengths = grid->wavelengths;

    // Loop over each particle
    for (int p = 0; p < npart; p++) {
        double mass = parts->masses[p];
        double fesc = parts->fesc[p];
        double velocity = part_velocities[p];  // Radial velocity for the particle

        // Compute Doppler shift factor
        double shift_factor = 1.0 + velocity / c;

        // Precompute shifted wavelengths
        double shifted_wavelengths[nlam];  // Array to hold shifted wavelengths
        for (int ilam = 0; ilam < nlam; ilam++) {
            shifted_wavelengths[ilam] = grid_wavelengths[ilam] * shift_factor;
        }

        // Now find the corresponding bin for each shifted wavelength
        for (int ilam = 0; ilam < nlam; ilam++) {
            double shifted_lambda = shifted_wavelengths[ilam];

            // Find the nearest wavelength bin for the shifted lambda
            int ilam_shifted = find_nearest_wavelength_bin(shifted_lambda, grid_wavelengths, nlam);

            // Check boundaries to avoid accessing out of bounds
            if (ilam_shifted >= nlam - 1) {
                ilam_shifted = nlam - 2;  // Use the last valid bin
            }

            // Interpolate the contribution between ilam_shifted and ilam_shifted + 1
            double frac_shifted = (shifted_lambda - grid_wavelengths[ilam_shifted]) /
                                  (grid_wavelengths[ilam_shifted + 1] - grid_wavelengths[ilam_shifted]);

            // Compute particle's contribution (adjusted for escape fraction)
            double weight = mass * (1.0 - fesc);

            // Distribute the spectra contribution across the nearest bins
            spectra[p * nlam + ilam_shifted] += (1.0 - frac_shifted) * weight;
            spectra[p * nlam + ilam_shifted + 1] += frac_shifted * weight;
        }
    }
}

int find_nearest_wavelength_bin(double lambda, double *grid_wavelengths, int nlam) {
    int nearest_index = 0;
    for (int ilam = 1; ilam < nlam; ilam++) {
        if (fabs(grid_wavelengths[ilam] - lambda) < fabs(grid_wavelengths[nearest_index] - lambda)) {
            nearest_index = ilam;
        }
    }
    return nearest_index;
}

int main() {
    // Define grid of wavelengths (in nm for example)
    double grid_wavelengths[N_LAM] = {400.0, 500.0, 600.0, 700.0, 800.0};  // nm
    Grid grid = {grid_wavelengths, N_LAM};

    // Define particle properties
    double masses[N_PARTICLES] = {1.0, 2.0, 0.5};  // Masses in arbitrary units
    double fesc[N_PARTICLES] = {0.1, 0.2, 0.15};  // Escape fractions
    double part_velocities[N_PARTICLES] = {1e7, -2e7, 5e6};  // Velocities in m/s

    Particles parts = {masses, fesc, N_PARTICLES};

    // Initialize spectra array
    double spectra[N_PARTICLES * N_LAM] = {0};  // Output array

    // Run the function
    spectra_loop_cic_serial(&grid, &parts, spectra, part_velocities);

    // Print the results
    printf("Spectra:\n");
    for (int p = 0; p < N_PARTICLES; p++) {
        printf("Particle %d:\n", p);
        for (int ilam = 0; ilam < N_LAM; ilam++) {
            printf("  Wavelength %f nm: %f\n", grid.wavelengths[ilam], spectra[p * N_LAM + ilam]);
        }
    }

    return 0;
}
