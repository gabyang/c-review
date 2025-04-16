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
    Params params{};
    std::vector<Particle> particles;
    read_args(argc, argv, params, particles);

    omp_set_num_threads(params.param_threads);

    int cellLen = params.param_radius * 6.5;
    int gridDim = params.square_size / cellLen + (params.square_size % cellLen > 0);

#if CHECK == 1
    SimulationValidator validator(params.param_particles, params.square_size, params.param_radius, params.param_steps);
    validator.initialize(particles);
#endif

    // Flatten grid and locks to 1D
    std::vector<omp_lock_t> locks(gridDim * gridDim);
    for (int i = 0; i < gridDim * gridDim; ++i)
        omp_init_lock(&locks[i]);

    std::vector<std::vector<int>> grid(gridDim * gridDim);
    std::vector<std::array<std::vector<std::pair<int, int>>, 5>> gridColPairs(gridDim * gridDim);
    std::vector<int> lastSet(gridDim * gridDim, -1);
    std::vector<Cell> partCells(params.param_particles);

    for (int step = 0; step < params.param_steps; step++) {
#pragma omp parallel for
        for (int i = 0; i < params.param_particles; i++) {
            particles[i].loc.x += particles[i].vel.x;
            particles[i].loc.y += particles[i].vel.y;
            partCells[i] = {std::clamp(static_cast<int>(floor(particles[i].loc.x / cellLen)), 0, gridDim - 1),
                            std::clamp(static_cast<int>(floor(particles[i].loc.y / cellLen)), 0, gridDim - 1)};
        }

#pragma omp parallel for
        for (int i = 0; i < params.param_particles; i++) {
            int y = partCells[i].y;
            int x = partCells[i].x;
            int cell_idx = y * gridDim + x;
            omp_set_lock(&locks[cell_idx]);
            if (lastSet[cell_idx] != step) {
                grid[cell_idx].clear();
                lastSet[cell_idx] = step;
            }
            grid[cell_idx].push_back(i);
            omp_unset_lock(&locks[cell_idx]);
        }

        bool hasOverlap = false;
#pragma omp parallel for reduction(|| : hasOverlap)
        for (int i = 0; i < gridDim; ++i) {
            for (int j = 0; j < gridDim; ++j) {
                int cell_idx = i * gridDim + j;
                for (auto& vec : gridColPairs[cell_idx]) vec.clear();

                for (size_t a = 0; a < grid[cell_idx].size(); ++a) {
                    if (i == 0 || i >= gridDim - 2 || j == 0 || j >= gridDim - 2) {
                        if (is_wall_overlap(particles[grid[cell_idx][a]].loc, params.square_size, params.param_radius)) {
                            hasOverlap = true;
                            gridColPairs[cell_idx][0].emplace_back(-1, grid[cell_idx][a]);
                        }
                    }
                    for (size_t b = a + 1; b < grid[cell_idx].size(); ++b) {
                        if (is_particle_overlap(particles[grid[cell_idx][a]].loc, particles[grid[cell_idx][b]].loc, params.param_radius)) {
                            hasOverlap = true;
                            gridColPairs[cell_idx][0].emplace_back(grid[cell_idx][a], grid[cell_idx][b]);
                        }
                    }
                }

                if (j < gridDim - 1) {
                    int right_idx = cell_idx + 1;
                    hasOverlap |= overlapHelper(particles, grid[cell_idx], grid[right_idx], params.param_radius, gridColPairs[cell_idx][1]);
                }
                if (i < gridDim - 1) {
                    int bottom_idx = cell_idx + gridDim;
                    hasOverlap |= overlapHelper(particles, grid[cell_idx], grid[bottom_idx], params.param_radius, gridColPairs[cell_idx][2]);
                }
                if (i < gridDim - 1 && j < gridDim - 1) {
                    int br_idx = cell_idx + gridDim + 1;
                    hasOverlap |= overlapHelper(particles, grid[cell_idx], grid[br_idx], params.param_radius, gridColPairs[cell_idx][3]);
                }
                if (i > 0 && j < gridDim - 1) {
                    int tr_idx = (i - 1) * gridDim + (j + 1);
                    hasOverlap |= overlapHelper(particles, grid[cell_idx], grid[tr_idx], params.param_radius, gridColPairs[cell_idx][4]);
                }
            }
        }

        if (hasOverlap) {
            bool hasCol;
            do {
                hasCol = false;

#pragma omp parallel for reduction(|| : hasCol)
                for (int cell_idx = 0; cell_idx < gridDim * gridDim; ++cell_idx) {
                    for (auto& [pai, pbi] : gridColPairs[cell_idx][0]) {
                        if (pai == -1) {
                            if (is_wall_collision(particles[pbi].loc, particles[pbi].vel, params.square_size, params.param_radius)) {
                                resolve_wall_collision(particles[pbi].loc, particles[pbi].vel, params.square_size, params.param_radius);
                                hasCol = true;
                            }
                        } else if (is_particle_moving_closer(particles[pai].loc, particles[pai].vel, particles[pbi].loc, particles[pbi].vel)) {
                            resolve_particle_collision(particles[pai].loc, particles[pai].vel, particles[pbi].loc, particles[pbi].vel);
                            hasCol = true;
                        }
                    }
                }

                // Check right neighbors
#pragma omp parallel for reduction(|| : hasCol)
                for (int i = 0; i < gridDim; ++i) {
                    for (int j = 0; j < gridDim - 1; ++j) {
                        int cell_idx = i * gridDim + j;
                        hasCol |= colhelper(particles, gridColPairs[cell_idx][1]);
                    }
                }

                // Check bottom neighbors
#pragma omp parallel for reduction(|| : hasCol)
                for (int j = 0; j < gridDim; ++j) {
                    for (int i = 0; i < gridDim - 1; ++i) {
                        int cell_idx = i * gridDim + j;
                        hasCol |= colhelper(particles, gridColPairs[cell_idx][2]);
                    }
                }

                // Check bottom-right neighbors
#pragma omp parallel for reduction(|| : hasCol)
                for (int i = 0; i < gridDim - 1; i += 2) {
                    for (int j = 0; j < gridDim - 1; ++j) {
                        int cell_idx = i * gridDim + j;
                        hasCol |= colhelper(particles, gridColPairs[cell_idx][3]);
                    }
                }
#pragma omp parallel for reduction(|| : hasCol)
                for (int i = 1; i < gridDim - 1; i += 2) {
                    for (int j = 0; j < gridDim - 1; ++j) {
                        int cell_idx = i * gridDim + j;
                        hasCol |= colhelper(particles, gridColPairs[cell_idx][3]);
                    }
                }

                // Check top-right neighbors
#pragma omp parallel for reduction(|| : hasCol)
                for (int i = 1; i < gridDim; i += 2) {
                    for (int j = 0; j < gridDim - 1; ++j) {
                        int cell_idx = i * gridDim + j;
                        hasCol |= colhelper(particles, gridColPairs[cell_idx][4]);
                    }
                }
#pragma omp parallel for reduction(|| : hasCol)
                for (int i = 2; i < gridDim; i += 2) {
                    for (int j = 0; j < gridDim - 1; ++j) {
                        int cell_idx = i * gridDim + j;
                        hasCol |= colhelper(particles, gridColPairs[cell_idx][4]);
                    }
                }

            } while (hasCol);
        }

#if CHECK == 1
        validator.validate_step(particles);
#endif
    }

    for (auto& lock : locks) omp_destroy_lock(&lock);
}