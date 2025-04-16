#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "collision.h"
#include "io.h"
#include "sim_validator.h"

bool colhelper(std::vector<Particle>& parts, std::vector<std::pair<int, int>>& cellOverlaps) {
    bool hasCol = false;
    for (auto& [pai, pbi] : cellOverlaps) {
        if (is_particle_moving_closer(parts[pai].loc, parts[pai].vel, parts[pbi].loc, parts[pbi].vel)) {
            hasCol = true;
            resolve_particle_collision(parts[pai].loc, parts[pai].vel, parts[pbi].loc, parts[pbi].vel);
        }
    }
    return hasCol;
}

bool overlapHelper(std::vector<Particle>& parts, std::vector<int>& cellA, std::vector<int>& cellB, int radius,
                   std::vector<std::pair<int, int>>& target) {
    int n = cellA.size(), m = cellB.size();
    bool hasOverlap = false;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (is_particle_overlap(parts[cellA[i]].loc, parts[cellB[j]].loc, radius)) {
                hasOverlap = true;
                target.push_back({cellA[i], cellB[j]});
            }
    return hasOverlap;
}

struct Cell {
    int x, y;
};

int main(int argc, char* argv[]) {
    // Read arguments and input file
    Params params{};
    std::vector<Particle> particles;
    read_args(argc, argv, params, particles);

    // Set number of threads
    omp_set_num_threads(params.param_threads);

    int cellLen = params.param_radius * 11;  // some eqn to scale up cellLen with density
    int gridDim = params.square_size / cellLen + (params.square_size % cellLen > 0);
    // std::cout << "cell len: " << cellLen << std::endl;
    // std::cout << "grid dim: " << gridDim << std::endl;

#if CHECK == 1
    // Initialize collision checker
    SimulationValidator validator(params.param_particles, params.square_size, params.param_radius, params.param_steps);
    // Initialize with starting positions
    validator.initialize(particles);
    // Uncomment the line below to enable visualization (makes program much slower)
    // validator.enable_viz_output("test.out");
#endif
    std::vector<std::vector<std::vector<int>>> grid(gridDim, std::vector<std::vector<int>>(gridDim));
    std::vector<std::vector<std::array<std::vector<std::pair<int, int>>, 5>>> gridColPairs(
        gridDim, std::vector<std::array<std::vector<std::pair<int, int>>, 5>>(gridDim));
    std::vector<Cell> partCells(params.param_particles);

    /**
        1. for movement of particles can be done in parallel
        2. wait for all particles to finish moving
        3. 3D vector grid[i][j] contains all particles at grid i,j
        Can be made parallel but requires lock on the shared grid as multiple particles can be in the same cell
        4. for each grid cell, check for collisions bet particles in cell then with adjacent cells, tr, r, br, b,
        repeat twice for tr and br each with 0 and 1 offset can be done in parallel as well
    **/
    for (int step = 0; step < params.param_steps; step++) {
#pragma omp parallel for
        for (int i = 0; i < params.param_particles; i++) {
            particles[i].loc.x += particles[i].vel.x;
            particles[i].loc.y += particles[i].vel.y;
            partCells[i] = {std::clamp(static_cast<int>(floor(particles[i].loc.x / cellLen)), 0, gridDim - 1),
                            std::clamp(static_cast<int>(floor(particles[i].loc.y / cellLen)), 0, gridDim - 1)};
        }

        // std::cout << "step: " << step << std::endl;
        // for (auto& part : particles) std::cout << part.loc.x << " " << part.loc.y << std::endl;

#pragma omp parallel for
        for (int i = 0; i < gridDim; i++)
            for (int j = 0; j < gridDim; j++) grid[i][j].clear();

        for (int i = 0; i < params.param_particles; i++) grid[partCells[i].y][partCells[i].x].push_back(i);

        bool hasOverlap = false;
// alternatively do all the checks here since gridColPairs has no conflicting write unlike resolve collision
#pragma omp parallel for reduction(|| : hasOverlap)
        for (int i = 0; i < gridDim; i++)
            for (int j = 0; j < gridDim; j++) {
                gridColPairs[i][j][0].clear();
                gridColPairs[i][j][1].clear();
                gridColPairs[i][j][2].clear();
                gridColPairs[i][j][3].clear();
                gridColPairs[i][j][4].clear();
                for (size_t a = 0; a < grid[i][j].size(); a++) {
                    if (i == 0 || i >= gridDim - 2 || j == 0 || j >= gridDim - 2)
                        if (is_wall_overlap(particles[grid[i][j][a]].loc, params.square_size, params.param_radius)) {
                            // std::cout << "part " << grid[i][j][a] << " collides with wall" << std::endl;
                            hasOverlap = true;
                            gridColPairs[i][j][0].push_back({-1, grid[i][j][a]});
                        }

                    for (size_t b = a + 1; b < grid[i][j].size(); b++) {
                        if (is_particle_overlap(particles[grid[i][j][a]].loc, particles[grid[i][j][b]].loc,
                                                params.param_radius)) {
                            hasOverlap = true;
                            gridColPairs[i][j][0].push_back({grid[i][j][a], grid[i][j][b]});
                        }
                    }
                }

                if (j < gridDim - 1)
                    hasOverlap |= overlapHelper(particles, grid[i][j], grid[i][j + 1], params.param_radius,
                                                gridColPairs[i][j][1]);
                if (i < gridDim - 1)
                    hasOverlap |= overlapHelper(particles, grid[i][j], grid[i + 1][j], params.param_radius,
                                                gridColPairs[i][j][2]);
                if (i < gridDim - 1 && j < gridDim - 1)
                    hasOverlap |= overlapHelper(particles, grid[i][j], grid[i + 1][j + 1], params.param_radius,
                                                gridColPairs[i][j][3]);
                if (i > 0 && j < gridDim - 1)
                    hasOverlap |= overlapHelper(particles, grid[i][j], grid[i - 1][j + 1], params.param_radius,
                                                gridColPairs[i][j][4]);
            }

        // std::cout << "hasOverlap " << (hasOverlap ? "true" : "false") << std::endl;
        if (hasOverlap) {
            while (true) {
                bool hasCol = false;

                // Check all cells in parallel (wall and particle collisions)
                #pragma omp parallel for reduction(||:hasCol)
                for (int cell = 0; cell < gridDim*gridDim; cell++) {
                    int i = cell / gridDim;
                    int j = cell % gridDim;
                    
                    // Check same-cell collisions
                    for (auto& [pai, pbi] : gridColPairs[i][j][0]) {
                        if (pai == -1) {  // Wall collision
                            if (is_wall_collision(particles[pbi].loc, particles[pbi].vel,
                                                params.square_size, params.param_radius)) {
                                resolve_wall_collision(particles[pbi].loc, particles[pbi].vel,
                                                    params.square_size, params.param_radius);
                                hasCol = true;
                            }
                        } else {  // Particle collision
                            if (is_particle_moving_closer(particles[pai].loc, particles[pai].vel,
                                                        particles[pbi].loc, particles[pbi].vel)) {
                                resolve_particle_collision(particles[pai].loc, particles[pai].vel,
                                                        particles[pbi].loc, particles[pbi].vel);
                                hasCol = true;
                            }
                        }
                    }
                }

                // Check all neighbor pairs in parallel (simplified neighbor checking)
                #pragma omp parallel for reduction(||:hasCol)
                for (int cell = 0; cell < gridDim*gridDim; cell++) {
                    int i = cell / gridDim;
                    int j = cell % gridDim;
                    
                    // Check right, bottom, and diagonal neighbors all at once
                    for (int dir = 1; dir <= 4; dir++) {
                        if (j < gridDim-1) hasCol |= colhelper(particles, gridColPairs[i][j][1]);  // Right
                        if (i < gridDim-1) hasCol |= colhelper(particles, gridColPairs[i][j][2]);  // Bottom
                        if (i < gridDim-1 && j < gridDim-1) hasCol |= colhelper(particles, gridColPairs[i][j][3]);  // BR
                        if (i > 0 && j < gridDim-1) hasCol |= colhelper(particles, gridColPairs[i][j][4]);  // TR
                    }
                }

                if (!hasCol) break;
            }
        }



#if CHECK == 1
        validator.validate_step(particles);
#endif
    }

}
