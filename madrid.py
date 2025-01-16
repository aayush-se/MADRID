import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
import os
from utils import contains_constant_regions, MASS_V2, nextpow2
from damp import DAMP_2_0
import time


def test_data_MADRID() -> np.ndarray:
    """Generates test data similar to the MATLAB implementation."""
    np.random.seed(123456789)
    fs = 10000
    t = np.arange(0, 10 + 1 / fs, 1 / fs)
    f_in_start = 50
    f_in_end = 60
    f_in = np.linspace(f_in_start, f_in_end, len(t))
    phase_in = np.cumsum(f_in / fs)
    y = np.sin(2 * np.pi * phase_in)
    y = y + np.random.randn(len(y)) / 12  # add noise

    end_of_train = len(y) // 2
    # Add medium anomaly
    y[end_of_train + 1200 : end_of_train + 1200 + 64] += np.random.randn(64) / 3
    # Add another medium anomaly
    y[end_of_train + 4180 : end_of_train + 4180 + 160] += np.random.randn(160) / 4
    # Add long anomaly
    y[end_of_train + 8200 : end_of_train + 8390] *= 0.5

    return y


def MADRID_2_0(
    T: np.ndarray,
    minL: int,
    maxL: int,
    stepSize: int,
    train_test_split: int,
    enable_output: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Runs MADRID but with adjusted parameters based on execution time. Additionally
    throws errors specific to the MADRID implementation.
    """
    pass


def MADRID(
    T: np.ndarray,
    minL: int,
    maxL: int,
    stepSize: int,
    train_test_split: int,
    enable_output: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Main MADRID algorithm implementation.

    Args:
        T: Input time series
        minL: Minimum subsequence length
        maxL: Maximum subsequence length
        stepSize: Step size for subsequence length
        train_test_split: Split point between training and testing data
        enable_output: Whether to show output plots

    Returns:
        Tuple containing:
        - MultiLengthDiscordTable: Discord scores for each length and position
        - BSF: Best-so-far scores for each length
        - BSF_loc: Locations of best-so-far scores
        - time_bf: Total computation time
    """
    BSFseed = -np.inf  # For first run of DAMP_topK
    k = 1
    time_bf = 0

    # Initialize arrays
    num_lengths = int(np.ceil((maxL + 1 - minL) / stepSize))
    MultiLengthDiscordTable = np.full((num_lengths, len(T)), -np.inf)
    BSF = np.zeros((num_lengths, 1))
    BSF_loc = np.full(num_lengths, np.nan)

    # For convergence plots
    time_sum_bsf = [[0, 0]]
    percent_sum_bsf = [[0, 0]]

    start_time = time.time()

    # Generate sequence lengths to test
    m_set = np.arange(minL, maxL, stepSize)
    m_pointer = len(m_set) // 2
    m = m_set[m_pointer]

    # Initial DAMP run
    left_mp, discord_score, position = DAMP_2_0(
        T, m, 1, train_test_split, enable_output=False
    )
    MultiLengthDiscordTable[m_pointer, :] = left_mp * (1 / (2 * np.sqrt(m)))
    BSF[m_pointer] = discord_score * (1 / (2 * np.sqrt(m)))
    BSF_loc[:] = position

    m_pointer = 0
    for m in m_set:

        if m_pointer == np.ceil(len(m_set) / 2):
            continue

        i = position
        sub_length = m
        if sub_length < 2 or i + sub_length - 1 > len(T):
            break

        query = T[i : i + sub_length - 1]

        # Use brute force to compute left MP
        MultiLengthDiscordTable[m_pointer, i] = np.min(
            np.real(MASS_V2(T[:i], query))
        ) * (1 / (2 * np.sqrt(m)))

        # Update the best so far discord score for current row
        BSF[m_pointer] = MultiLengthDiscordTable[m_pointer, i]
        BSF_loc[m_pointer] = i
        m_pointer += 1

    m_pointer = 0
    m = m_set[m_pointer]
    left_mp, _, position_2 = DAMP_2_0(T, m, 1, train_test_split, enable_output=False)
    MultiLengthDiscordTable[m_pointer, :] = left_mp * (1 / (2 * np.sqrt(m)))

    BSF[m_pointer] = np.max(MultiLengthDiscordTable[m_pointer, :])
    BSF_loc[m_pointer] = np.argmax(MultiLengthDiscordTable[m_pointer, :])

    if position_2 != position:
        m_pointer = 0
        for m in m_set:

            if m_pointer == np.ceil(len(m_set) / 2) or m_pointer == 1:
                continue

            i = position_2
            sub_length = m
            if i + sub_length - 1 > len(T):
                break

            query = T[i : i + sub_length - 1]
            MultiLengthDiscordTable[m_pointer, i] = np.min(
                np.real(MASS_V2(T[:i], query))
            ) * (1 / (2 * np.sqrt(m)))

            BSF[m_pointer] = np.max(MultiLengthDiscordTable[m_pointer, :])
            BSF_loc[m_pointer] = np.argmax(MultiLengthDiscordTable[m_pointer, :])
            m_pointer += 1

    m_pointer = len(m_set) - 1
    m = m_set[m_pointer]
    left_mp, _, position_3 = DAMP_2_0(T, m, 1, train_test_split, enable_output=False)
    MultiLengthDiscordTable[m_pointer, :] = left_mp * (1 / (2 * np.sqrt(m)))

    BSF[m_pointer] = np.max(MultiLengthDiscordTable[m_pointer, :])
    BSF_loc[m_pointer] = np.argmax(MultiLengthDiscordTable[m_pointer, :])

    if position_3 != position_2 and position_3 != position:
        m_pointer = 0
        for m in m_set:

            if m_pointer == np.ceil(len(m_set) / 2) or m_pointer == 1:
                continue

            i = position_3
            sub_length = m
            if i + sub_length - 1 > len(T):
                break

            query = T[i : i + sub_length - 1]

            # Use brute force to compute left MP
            MultiLengthDiscordTable[m_pointer, i] = np.min(
                np.real(MASS_V2(T[0:i], query))
            ) * (1 / (2 * np.sqrt(m)))

            # Update the best so far discord score for current row
            BSF[m_pointer] = np.max(MultiLengthDiscordTable[m_pointer, :])
            BSF_loc[m_pointer] = np.argmax(MultiLengthDiscordTable[m_pointer, :])
            m_pointer += 1

    # Update data for storage plots
    # initialization_time = time.time() - start_time
    # time_bf += initialization_time
    # time_sum_bsf.append([time_bf, time_bf])
    # percent_sum_bsf.append([time_bf / time_bf, time_bf / time_bf])

    m_pointer = 0
    for m in m_set:

        if (
            m_pointer == np.ceil(len(m_set) / 2)
            or m_pointer == 1
            or m_pointer == len(m_set)
        ):
            continue

        Results, BFS_for_i_plus_1, left_mp = DAMP_topK_new(
            T,
            train_test_split,
            m,
            k,
            False,
            max(BSFseed, BSF[m_pointer]),
        )

        BSF[m_pointer] = Results[0, 0] * (1 / (2 * np.sqrt(m)))
        BSF_loc[m_pointer] = Results[0, 1]  # Only for k = 1

        MultiLengthDiscordTable[m_pointer, :] = left_mp * (1 / (2 * np.sqrt(m)))

        BSFseed = BFS_for_i_plus_1 - 0.000001
        m_pointer += 1

    return MultiLengthDiscordTable, BSF, BSF_loc


def DAMP_topK_new(
    T: np.ndarray,
    CurrentIndex: int,
    SubsequenceLength: int,
    discord_num: int,
    enable_output: bool,
    BSFseed: float,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Special Matrix Profile implementation that only looks backwards in time.

    Args:
        T: Input time series
        CurrentIndex: Starting index for processing
        SubsequenceLength: Length of subsequences to compare
        discord_num: Number of discords to find
        enable_output: Whether to show output
        BSFseed: Initial best-so-far value

    Returns:
        Results: Array of discord scores and positions
        BFS_for_i_plus_1: Best-so-far value for next iteration
        Left_MP: Left Matrix Profile values
    """
    # Initialize left matrix profile
    Left_MP = np.zeros(len(T))

    # Initialize parameters
    best_so_far = BSFseed
    bool_vec = np.ones(len(T))
    lookahead = int(2 ** nextpow2(16 * SubsequenceLength))

    # Main loop
    for i in range(CurrentIndex, len(T) - SubsequenceLength + 1):
        # Skip if boolean is 0
        if not bool_vec[i]:
            Left_MP[i] = Left_MP[i - 1] - 0.00001
            continue

        # Break if beyond time series
        if i + SubsequenceLength - 1 > len(T):
            break

        # Initialize DAMP parameters
        approximate_distance = float("inf")
        X = int(2 ** nextpow2(8 * SubsequenceLength))
        flag = True
        expansion_num = 0
        query = T[i : i + SubsequenceLength - 1]

        # Classic DAMP
        while approximate_distance >= best_so_far:
            # Case 1: At beginning of time series
            if i - X + 1 + (expansion_num * SubsequenceLength) < 1:
                approximate_distance = np.min(np.real(MASS_V2(T[0:i], query)))
                Left_MP[i] = approximate_distance

                # Update best discord
                if approximate_distance > best_so_far:
                    best_so_far = approximate_distance
                    Left_MP_copy = Left_MP.copy()
                    for k in range(discord_num):
                        best_so_far = np.max(Left_MP_copy)
                        idx_max = np.argmax(Left_MP_copy)
                        discord_start = max(0, idx_max - (SubsequenceLength // 2))
                        discord_end = max(
                            1 + (SubsequenceLength // 2),
                            idx_max + (SubsequenceLength // 2),
                        )
                        Left_MP_copy[discord_start:discord_end] = float("-inf")
                break

            else:
                if flag:
                    # Case 2: Closest segment
                    flag = False
                    approximate_distance = np.min(
                        np.real(MASS_V2(T[i - X + 1 : i], query))
                    )
                else:
                    # Case 3: Other cases
                    X_start = i - X + 1 + (expansion_num * SubsequenceLength)
                    X_end = i - (X // 2) + (expansion_num * SubsequenceLength)
                    approximate_distance = np.min(
                        np.real(MASS_V2(T[X_start:X_end], query))
                    )

                if approximate_distance < best_so_far:
                    Left_MP[i] = approximate_distance
                    break
                else:
                    X *= 2
                    expansion_num += 1

        # Commented out as we are focused on the Online MADRID implementation
        # Forward pruning if lookahead enabled
        # if lookahead != 0:
        #     start_of_mass = min(i + SubsequenceLength, len(T))
        # end_of_mass = min(start_of_mass + lookahead - 1, len(T))

        # if (end_of_mass - start_of_mass + 1) > SubsequenceLength:
        #     distance_profile = np.real(MASS_V2(T[start_of_mass:end_of_mass], query))
        #     dp_index_less_than_BSF = np.where(distance_profile < best_so_far)[0]
        #     ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass
        #     bool_vec[ts_index_less_than_BSF] = 0

    # Get results
    Results = np.zeros((discord_num, 2))
    BFS_for_i_plus_1 = []

    # Calculate pruning rate
    PV = bool_vec[CurrentIndex : len(T) - SubsequenceLength + 1]
    PR = (len(PV) - np.sum(PV)) / len(PV)
    if enable_output:
        print(f"Pruning Rate: {PR}")

    # Find top K discords
    Left_MP_copy = Left_MP.copy()
    for k in range(discord_num):
        val = np.max(Left_MP_copy)
        loc = np.argmax(Left_MP_copy)

        if val == 0:
            if enable_output:
                print(f"Only {k-1} discords are found.")
            if k == 0:
                BFS_for_i_plus_1.append(float("-inf"))
            break

        if enable_output:
            print(f"Predicted discord score/position (top {k+1}): {val}/{loc}")

        Results[k] = [val, loc]

        discord_start = max(0, loc)
        discord_end = max(1 + SubsequenceLength + 1, loc + SubsequenceLength + 1)
        BFS_for_i_plus_1.append(
            np.min(
                np.real(
                    MASS_V2(
                        T[0:discord_start],
                        T[discord_start : min(discord_start * 2 - 1, discord_end)],
                    )
                )
            )
        )
        Left_MP_copy[discord_start:discord_end] = float("-inf")

    BFS_for_i_plus_1 = min(BFS_for_i_plus_1)

    if enable_output:
        plt.figure()
        plt.plot(Left_MP, "b")
        plt.plot((T - np.mean(T)) / np.std(T) - 2, "r")
        plt.show()

    return Results, BFS_for_i_plus_1, Left_MP


if __name__ == "__main__":
    # Generate test data
    time_series = test_data_MADRID()

    # Run MADRID
    MultiLengthDiscordTable, BSF, BSF_loc = MADRID(
        T=time_series[:3000],
        minL=50,
        maxL=200,
        stepSize=10,
        train_test_split=672,
        enable_output=True,
    )
