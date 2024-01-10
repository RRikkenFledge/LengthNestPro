import copy
import math
import CalculationParameters

import numpy as np

from column_sort import ColumnSorter


class CalculateObject:
    calcParams: CalculationParameters

    def __init__(self):
        self.calculation_was_canceled = 0
        self.calcParams = None

    # TODO add algorithm/option that tends to pick nests that use up a single part faster (reduces the dependency of the
    #  patterns on the part quantities)

    # TODO add algorithm/option to reduce the number of containers needed (completely finish first 3 parts before
    #  starting on the next parts.  Never cut the nth part if the (num_parts-3)th part is not completed.  Allow user
    #  to adjust 3 to other values.

    # TODO add functionality to calculate optimum stock length (by iterating with different stock lengths?)

    # TODO allow user to select multiple stock lengths and quantities/priorities

    # TODO remove timers

    # TODO add file about for version info and help

    # Create function to nest parts
    def length_nest_pro_calculate(self, calc_params: CalculationParameters):
        self.calcParams = calc_params
        warning_messages = []
        ##########################
        # Pre-processing section #
        ##########################

        # Start timer
        # nesting_start_time = time.time()

        max_iterations: int = 1000000

        # Check how many different parts are needed (num_parts)
        num_parts: int = len(self.calcParams.part_lengths)

        # Set precision for printing
        np.set_printoptions(precision=3)

        # Remove any parts where the part quantity is 0 (remove entries from self.calcParams.part_names,
        # self.calcParams.part_lengths, and self.calcParams.part_quantities)
        for i in range(len(self.calcParams.part_quantities))[::-1]:
            if self.calcParams.part_quantities[i] == 0:
                self.calcParams.part_quantities = np.delete(self.calcParams.part_quantities, i, 0)
                self.calcParams.part_lengths = np.delete(self.calcParams.part_lengths, i, 0)
                self.calcParams.part_names = np.delete(self.calcParams.part_names, i, 0)

        initial_part_quantities = self.calcParams.part_quantities.copy()
        initial_part_lengths = self.calcParams.part_lengths.copy()
        initial_part_names = self.calcParams.part_names.copy()

        # Update number of parts after removing parts with qty 0
        num_parts: int = len(self.calcParams.part_lengths)

        # Make sure max_containers is not higher than num_parts, and make sure it was not a string.
        if self.calcParams.max_containers > num_parts or self.calcParams.max_containers == -2:
            self.calcParams.max_containers = num_parts

        # Make sure max_parts_per_nest is not greater than max_containers since that wouldn't make sense.
        # Also make sure it was not entered as a string
        if self.calcParams.max_parts_per_nest > self.calcParams.max_containers or self.calcParams.max_parts_per_nest == -2:
            self.calcParams.max_parts_per_nest = self.calcParams.max_containers

        # TODO Combine parts with same length?  Must still consider quantities...

        # TODO Find all optimum nodes, not just one

        # Solve for nestable length with extra spacing adjustment since blank includes spacing
        initial_nestable_length = \
            self.calcParams.stock_length \
            - self.calcParams.left_waste \
            - self.calcParams.right_waste \
            + self.calcParams.spacing

        nestable_length = initial_nestable_length

        # Construct length vector by adding part spacing
        nested_lengths = np.zeros((num_parts, 1))
        for i in range(num_parts):
            nested_lengths[i, 0] = self.calcParams.part_lengths[i, 0] + self.calcParams.spacing

        initial_nested_lengths = nested_lengths.copy()

        # Initialize patterns matrix (Step #1) (single part nesting patterns)
        patterns = np.zeros((num_parts, num_parts))
        for i in range(num_parts):
            patterns[i, i] = math.floor(nestable_length / nested_lengths[i])

            # Check if each part can be nested on the available length
            if (patterns[i, i] == 0
                    or self.calcParams.stock_length < self.calcParams.left_waste + self.calcParams.right_waste):
                # Zero out all outputs and exit function
                final_patterns = []
                final_allocations = 0
                self.calcParams.error = 1  # error code 1 signifies that a part is too long

                return final_patterns, final_allocations

        # Find required number of lengths if everything nests ideally (only possible if parts nest perfectly on nestable
        # length)
        ideal_num = (np.dot(np.transpose(nested_lengths),
                            self.calcParams.part_quantities) / nestable_length).item()

        warning_messages.append(
            "Ideally, the job would only require " + str(round(ideal_num, 2)) + " lengths. (zero scrap)"
        )

        # Find required number of lengths in worst case scenario (single part nests only)
        patterns_inv = np.linalg.inv(patterns)
        patterns_trans_inv = np.transpose(patterns_inv)
        ones_vector = np.ones((num_parts, 1))

        # pi is a measure of how much of a stock length is used to cut a given part (if the 3rd term is 0.25, that
        # would indicate that the third part uses 1/4 of a stock length when considering the entire nest with all parts)
        pi = np.dot(patterns_trans_inv, ones_vector)
        worst_case = np.dot(np.transpose(self.calcParams.part_quantities), pi)
        required_lengths = worst_case.copy()

        warning_messages.append(
            "If only single part nests are used, the job would require a maximum of " + str(
                round(worst_case.item(), 2))
            + " lengths."
        )

        # Initialize parts_sublist (will restrain column generation to only consider parts spanning range of
        # max_containers)
        parts_sublist = np.zeros((num_parts - self.calcParams.max_containers + 1, self.calcParams.max_containers))
        parts_sublist_sorted = np.zeros((1, num_parts))
        container_counter = 0
        remaining_iterations = -1

        # TODO remove this feature if it is not desired (speeds up convergence, but results are less consistent)
        # # Initialize the part ordering if it doesn't exist yet
        # try:
        #     print(self.calcParams.current_sequence)
        # except AttributeError:
        #     self.calcParams.current_sequence = np.array(range(self.num_parts))

        # Initialize the part ordering
        self.calcParams.current_sequence = np.arange(num_parts)

        for sub_i in range(num_parts - self.calcParams.max_containers + 1):
            parts_sublist[sub_i] = np.array(range(sub_i, sub_i + self.calcParams.max_containers))
            # self.parts_sublist[sub_i] = self.calcParams.current_sequence[sub_i:(sub_i +
            # self.calcParams.max_containers)]

        # Iterate the part sequence to find the part ordering for which scrap is minimized
        part_sequence_is_optimum = 0
        current_part_index = 0

        self.calcParams.part_quantities = initial_part_quantities[self.calcParams.current_sequence].copy()
        self.calcParams.part_lengths = initial_part_lengths[self.calcParams.current_sequence].copy()
        self.calcParams.part_names = initial_part_names[self.calcParams.current_sequence].copy()

        parts_sublist_sorted = np.zeros((
            num_parts - self.calcParams.max_containers + 1,
            self.calcParams.max_containers
        ))

        best_sequence = self.calcParams.current_sequence.copy()

        # Run through one more time using best sequence to get best patterns
        self.calcParams.part_quantities = initial_part_quantities[best_sequence].copy()
        self.calcParams.part_lengths = initial_part_lengths[best_sequence].copy()
        self.calcParams.part_names = initial_part_names[best_sequence].copy()

        nested_lengths = initial_nested_lengths[best_sequence].copy()

        # Reinitialize nestable length
        nestable_length = initial_nestable_length

        # Reinitialize patterns matrix (single part nesting patterns)
        patterns = np.zeros((num_parts, num_parts))
        for k in range(num_parts):
            patterns[k, k] = math.floor(nestable_length / nested_lengths[k])

        loop_count = None
        index_order = None
        ip_best = None
        ip_best_row = None
        # Run the column generation algorithm to find the best set of nest patterns for the job quantities
        # Decrease max parts per nest until max containers is fulfilled.
        for i in range(1, 1 + self.calcParams.max_parts_per_nest)[::-1]:
            temp_max_parts_per_nest = i
            [required_lengths, allocation, patterns, loop_count, index_order, ip_best, ip_best_row] = self.column_gen(
                num_parts,
                temp_max_parts_per_nest,
                self.calcParams.part_quantities,
                np.array([np.arange(num_parts)]),
                0,
                num_parts,
                nestable_length,
                nested_lengths,
                max_iterations,
                parts_sublist_sorted,
                loop_count,
                index_order,
                ip_best,
                ip_best_row
            )

            # Check if calculation has been canceled.
            if self.calculation_was_canceled == 1:
                # Zero out all outputs and exit function
                final_patterns = []
                final_allocations = 0

                return final_patterns, final_allocations

            # Round all the allocations down, but allow for rounding error
            int_allocation = allocation.copy()
            for ii in range(len(allocation))[::-1]:
                int_allocation[ii] = math.floor(allocation[ii] + 0.0000001)
                # Remove unused patterns
                if int_allocation[ii] == 0:
                    int_allocation = np.delete(int_allocation, ii, 0)
                    allocation = np.delete(allocation, ii, 0)
                    patterns = np.delete(patterns.T, ii, 0).T

            # TODO move this to a better place
            column_sort = ColumnSorter(num_parts, len(allocation), self)

            if patterns.any():
                [new_column_order, required_containers] = column_sort.optimize_sequence(patterns, 100, 0)

                # TODO remove second condition later?
                if required_containers <= self.calcParams.max_containers or temp_max_parts_per_nest == 2:
                    patterns = patterns.T[new_column_order].T
                    int_allocation = int_allocation[new_column_order]

                    chosen_required_lengths = required_lengths.copy()
                    chosen_int_allocation = int_allocation.copy()
                    chosen_patterns = patterns.copy()
                    warning_messages.append(
                        f"Acceptable solution found with 'Max parts per nest' constrained to {i}"
                    )
                    warning_messages.append(f"Requires about {required_lengths} lengths")

                    break
            else:
                chosen_required_lengths = required_lengths.copy()
                chosen_int_allocation = int_allocation.copy()
                chosen_patterns = patterns.copy()
                warning_messages.append(f"No integer solution was found with 'Max parts per nest' constrained to {i}")

                break

        # TODO check for best solution before filling remaining parts

        required_lengths = chosen_required_lengths.copy()
        int_allocation = chosen_int_allocation.copy()
        patterns = chosen_patterns.copy()

        # Create active parts matrix, and find start_pattern and end_pattern for each part inside pre_process function
        active_parts = patterns.copy()
        active_parts = column_sort.pre_process(active_parts)

        # Sort parts (rows) by end_pattern (earliest to latest)
        sorted_by_start = np.argsort(column_sort.start_pattern)
        column_sort.end_pattern_sorted = column_sort.end_pattern[sorted_by_start]
        column_sort.end_pattern_sorted = column_sort.end_pattern_sorted.astype(float)
        for item_index, item in enumerate(column_sort.end_pattern_sorted):
            column_sort.end_pattern_sorted[item_index] = item + item_index * 0.0000000001
        sorted_by_end = np.argsort(column_sort.end_pattern_sorted)
        active_parts = active_parts[sorted_by_start][sorted_by_end]
        patterns = patterns[sorted_by_start][sorted_by_end]
        self.calcParams.part_quantities = self.calcParams.part_quantities[sorted_by_start][sorted_by_end]
        self.calcParams.part_names = self.calcParams.part_names[sorted_by_start][sorted_by_end]
        self.calcParams.part_lengths = self.calcParams.part_lengths[sorted_by_start][sorted_by_end]
        nested_lengths = nested_lengths[sorted_by_start][sorted_by_end]
        column_sort.start_pattern = column_sort.start_pattern[sorted_by_start][sorted_by_end]
        column_sort.end_pattern = column_sort.end_pattern[sorted_by_start][sorted_by_end]
        frozen_pi = pi[sorted_by_start][sorted_by_end]

        containers = column_sort.count_containers(active_parts)
        containers_equals_max = containers == self.calcParams.max_containers

        # Subtract quantities of fully allocated sticks to find remaining_part_quantities
        nested = np.dot(patterns, int_allocation)
        remaining_part_quantities = self.calcParams.part_quantities.copy()
        remaining_part_quantities = remaining_part_quantities - nested

        # Initialize with 2 dummy terms
        additional_patterns = np.zeros((num_parts, 2))
        additional_allocations = np.zeros((1, 2))

        # TODO analyze how well this bug fix works and improve
        #  maybe try simple allocation without nesting
        # Fix cs.start_pattern and cs.end_pattern if they contain terms that are out of bounds
        acceptable_values = np.arange(len(int_allocation))
        skip_to_end = 0
        for num_index, num in enumerate(column_sort.start_pattern):
            if num not in acceptable_values:
                skip_to_end = 1

        # Initialize with dummy term
        bridge_position_tracker = np.array([[0]])

        # Loop through parts (top to bottom)
        for part_index, active_parts_row in enumerate(active_parts):
            # If part is unfulfilled, generate a bridge pattern to fulfill it without breaking max_containers req
            # Use the nest with the lowest possible scrap rate (greedy algorithm),
            #   but add preference for fulfilling other unfulfilled parts
            # Stop early if remaining parts can be nested without breaking max_containers requirement
            if part_index >= (num_parts - self.calcParams.max_containers):
                break
            if skip_to_end == 1:
                break
            if remaining_part_quantities[part_index]:
                # Find index of next unfulfilled part after current part
                next_uff_part = num_parts - 1
                for ii in range(part_index + 1, num_parts):
                    if remaining_part_quantities[ii]:
                        next_uff_part = ii
                        break

                # Define left and right bounds for current parts
                left_bound = int(column_sort.start_pattern[part_index])
                right_bound = int(column_sort.end_pattern[next_uff_part])

                # Adjust right bound to avoid breaking max_containers requirement
                for j in range(left_bound, right_bound):
                    # Using j as the left pattern under the below conditions would break the max_containers requirement
                    if containers_equals_max[j] and active_parts_row[j] == 0:
                        right_bound = j
                        break

                if left_bound == right_bound:
                    left_is_same_as_right = 1
                else:
                    left_is_same_as_right = 0

                # Loop through pairs of patterns within bounds to define parts that may be included in bridge patterns
                bridge_sublist = np.zeros((right_bound + left_is_same_as_right - left_bound, num_parts))
                for j_index, j in enumerate(range(left_bound, right_bound + left_is_same_as_right)):
                    left_active = active_parts[:, j]
                    right_active = active_parts[:, j + 1 - left_is_same_as_right]

                    # Current part should always be used in bridge pattern
                    bridge_sublist[j_index][part_index] = 1

                    # Other parts are allowed in bridge pattern if they are adjacent to an active part
                    # Other parts must be below current part
                    for ii in range(part_index + 1, num_parts):
                        if left_active[ii] or right_active[ii]:
                            bridge_sublist[j_index][ii] = 1

                # bridge_sublist = np.unique(bridge_sublist, axis=0)
                # print(bridge_sublist)  # TODO add similar functionality later

                unavailable_parts = [[-1]]
                try_expanding_right_bound = 0
                num_extra_parts = 0

                while remaining_part_quantities[part_index]:  # Finds additional bridge pattern if needed
                    if try_expanding_right_bound == 1 and right_bound < np.shape(active_parts)[1] - 2:
                        right_bound += 1
                        bridge_sublist = np.append(bridge_sublist, np.zeros((1, num_parts)), axis=0)

                        # Loop through pairs of adjacent patterns within bounds to define parts that may be included in
                        #   each bridge patterns
                        left_active = active_parts[:, right_bound]
                        right_active = active_parts[:, right_bound + 1]

                        # Current part should always be used in bridge pattern
                        bridge_sublist[-1][part_index] = 1

                        # Other parts are allowed in bridge pattern if they are adjacent to an active part
                        # Other parts must be below current part
                        for ii in range(part_index + 1, num_parts):
                            if left_active[ii] or right_active[ii]:
                                bridge_sublist[j_index][ii] = 1

                    best_bridge_pattern = []
                    best_bridge_ip = 0
                    for sublist_index, sublist in enumerate(bridge_sublist):
                        parts_sublist = np.array([])
                        for bit_index, bit in enumerate(sublist):
                            if bit == 1:
                                parts_sublist = np.append(parts_sublist, bit_index)
                        parts_sublist = np.array([parts_sublist.astype(int)])
                        container_counter = 0

                        # Create new sublist that helps branch bound function decide which nests are best
                        bonus_sublist = parts_sublist.copy()
                        for item_index, item in enumerate(bonus_sublist[0]):
                            if item == part_index or item == next_uff_part:
                                bonus_sublist[0][item_index] = 1
                            elif item > next_uff_part:
                                bonus_sublist[0][item_index] = 0
                            else:
                                bonus_sublist[0][item_index] = -1

                        ip_best, ip_best_row, index_order = self.branch_bound(
                            len(parts_sublist[0]),
                            self.calcParams.max_parts_per_nest + num_extra_parts,
                            self.calcParams.part_quantities.copy(), parts_sublist, 1,
                            np.where(parts_sublist[0] == part_index)[0].item(),
                        )

                        # Check if calculation has been canceled.
                        if self.calculation_was_canceled == 1:
                            # Zero out all outputs and exit function
                            final_patterns = []
                            final_allocations = 0

                            return final_patterns, final_allocations

                        # Extract the best pattern from ip_best_row
                        best_pattern_sorted_sublist = \
                            np.transpose([ip_best_row[1:(len(parts_sublist[0]) + 1)]])

                        # Initialize best_pattern_sorted
                        best_pattern_sorted = np.zeros((num_parts, 1))

                        # Add pattern values from the sublist to the main list
                        for iii in range(len(parts_sublist[0])):
                            # Find index in main list corresponding to ith index of parts_sublist_sorted
                            corr_index = \
                                np.where(index_order == parts_sublist_sorted[container_counter][iii])[
                                    0].item()
                            best_pattern_sorted[corr_index] = best_pattern_sorted_sublist[iii]

                        # Reorder best_pattern vector
                        old_index_order = np.argsort(index_order)
                        best_pattern = best_pattern_sorted[old_index_order]

                        if ip_best > best_bridge_ip:
                            best_bridge_ip = ip_best
                            best_bridge_pattern = best_pattern.copy()
                            insertion_position = column_sort.start_pattern[part_index] + sublist_index + 1

                    # Check if any downstream unfulfilled parts could have fit on current bridge pattern
                    remaining_length = (
                            self.calcParams.stock_length - self.calcParams.left_waste - self.calcParams.right_waste
                    )
                    remaining_length -= np.dot(best_bridge_pattern.T, nested_lengths)

                    downstream_parts = np.arange(part_index + 1, num_parts)

                    find_new_best = 0
                    for p_index, part in enumerate(downstream_parts):
                        if not best_bridge_pattern[part][0]:  # If the part is not in the bridge pattern
                            if nested_lengths[p_index] < remaining_length:  # and if there is room to nest it
                                if right_bound < np.shape(active_parts)[1] - 2:  # and the right bound can be
                                    # extended
                                    try_expanding_right_bound = 1
                                    find_new_best = 1
                                    num_parts_in_nest = 0
                                    for p in best_bridge_pattern:
                                        if p[0]:
                                            num_parts_in_nest += 1
                                    if num_parts_in_nest == self.calcParams.max_parts_per_nest + num_extra_parts:
                                        num_extra_parts += 1
                                    break

                    if find_new_best == 1:
                        continue
                    else:
                        num_extra_parts = 0

                    # Save copies of changing variables in case they have to change back
                    backup_vars = [
                        bridge_position_tracker.copy(),
                        additional_patterns.copy(),
                        additional_allocations.copy(),
                        remaining_part_quantities.copy(),
                        int_allocation.copy()
                    ]

                    bridge_position_tracker = np.append(bridge_position_tracker, insertion_position)
                    additional_patterns = np.append(additional_patterns.T, best_bridge_pattern.T, axis=0).T
                    additional_allocations = np.append(additional_allocations.T, [[1]], axis=0).T
                    remaining_part_quantities -= best_bridge_pattern

                    do_not_adjust_allocations = 0
                    # Loop to decide which patterns to borrow from
                    while np.any(remaining_part_quantities < 0):
                        for pattern_qty_index, pattern_qty in enumerate(best_bridge_pattern):
                            if pattern_qty:
                                if pattern_qty_index <= part_index:  # Catches part_index part
                                    continue
                                # Borrow from the rightmost possible pattern if there are negative parts remaining
                                if remaining_part_quantities[pattern_qty_index] < 0:
                                    rightmost_pattern_index = column_sort.end_pattern[pattern_qty_index]
                                    # Make sure there are patterns available to pull from before borrowing parts
                                    if int_allocation[rightmost_pattern_index] == 0:
                                        # Keep trying patterns to the left until there are allocations for that pattern,
                                        # and it uses the part with negative qty remaining
                                        while int_allocation[rightmost_pattern_index] == 0 or \
                                                patterns[pattern_qty_index][rightmost_pattern_index] == 0:
                                            rightmost_pattern_index = rightmost_pattern_index - 1

                                            # TODO come up with a better solution here
                                            if rightmost_pattern_index == -1:
                                                # Something must be wrong in code if this happens, but skipping to end
                                                #   should avoid a crash
                                                skip_to_end = 1
                                            else:
                                                for qty_index, qty in \
                                                        enumerate(patterns.T[rightmost_pattern_index]):
                                                    if qty:
                                                        highest_in_pattern = qty_index
                                                        break

                                            # If pattern contains any parts with index <= self.part_index (higher
                                            # than current part)
                                            if skip_to_end == 1 or highest_in_pattern <= part_index:
                                                # Restore variables if no downstream parts can be borrowed
                                                [
                                                    bridge_position_tracker,
                                                    additional_patterns,
                                                    additional_allocations,
                                                    remaining_part_quantities,
                                                    int_allocation
                                                ] = backup_vars.copy()

                                                unavailable_parts = np.append(
                                                    unavailable_parts,
                                                    [[pattern_qty_index]],
                                                    axis=0
                                                )
                                                do_not_adjust_allocations = 1

                                                break

                                    if do_not_adjust_allocations == 0:
                                        int_allocation[rightmost_pattern_index] -= 1
                                        remaining_part_quantities += np.expand_dims(
                                            patterns.T[rightmost_pattern_index],
                                            axis=0
                                        ).T

                                    if skip_to_end == 1:
                                        break

                    if skip_to_end == 1:
                        break

            if skip_to_end == 1:
                break

        # Delete dummy term
        bridge_position_tracker = np.delete(bridge_position_tracker, 0, axis=0)

        if sum(remaining_part_quantities) == 0:
            parts_need_nested = 0
        else:
            parts_need_nested = 1

        num_extra_parts = 0

        while parts_need_nested == 1:

            final_parts_sublist = np.arange(num_parts - self.calcParams.max_containers, num_parts)
            if skip_to_end == 1:
                final_parts_sublist = np.arange(num_parts)
            final_parts_sublist = np.expand_dims(final_parts_sublist, axis=0)
            if skip_to_end == 1:
                self.branch_bound(num_parts, self.calcParams.max_parts_per_nest + num_extra_parts,
                                  remaining_part_quantities, final_parts_sublist, 2)
            else:
                self.branch_bound(self.calcParams.max_containers, self.calcParams.max_parts_per_nest + num_extra_parts,
                                  remaining_part_quantities, final_parts_sublist, 2)

            # Check if calculation has been canceled.
            if self.calculation_was_canceled == 1:
                # Zero out all outputs and exit function
                final_patterns = []
                final_allocations = 0

                return final_patterns, final_allocations

            # Extract the best pattern from ip_best_row
            best_pattern_sorted_sublist = np.transpose([ip_best_row[1:(len(ip_best_row) - 3)]])

            # Initialize best_pattern_sorted
            best_pattern_sorted = np.zeros((num_parts, 1))

            # Add pattern values from the sublist to the main list
            for iii in range(len(ip_best_row) - 4):
                # Find index in main list corresponding to ith index of parts_sublist_sorted
                corr_index = \
                    np.where(index_order == parts_sublist_sorted[container_counter][iii])[
                        0].item()
                best_pattern_sorted[corr_index] = best_pattern_sorted_sublist[iii]

            # Reorder best_pattern vector
            old_index_order = np.argsort(index_order)
            best_pattern = best_pattern_sorted[old_index_order]

            # TODO add above to branch_bound function? see other occurances

            # Check if any downstream unfulfilled parts could have fit on current bridge pattern
            remaining_length = self.calcParams.stock_length - self.calcParams.left_waste - self.calcParams.right_waste
            remaining_length -= np.dot(best_pattern.T, nested_lengths)

            downstream_parts = np.arange(part_index, num_parts)

            find_new_best = 0
            for p_index, part in enumerate(downstream_parts):
                if nested_lengths[p_index] < remaining_length:  # check if there is room to nest it
                    if self.calcParams.max_parts_per_nest + num_extra_parts <= len(downstream_parts):
                        find_new_best = 1
                        num_extra_parts += 1
                        break

            if find_new_best == 1:
                continue
            else:
                num_extra_parts = 0

            additional_patterns = np.append(additional_patterns.T, best_pattern.T, axis=0).T
            additional_allocations = np.append(additional_allocations.T, [[1]], axis=0).T

            remaining_part_quantities -= best_pattern

            # While loop to add additional patterns with remaining parts
            too_many_nested = 0
            while too_many_nested == 0 and parts_need_nested == 1:
                check_totals = remaining_part_quantities - best_pattern

                for i in range(num_parts):
                    if check_totals[i] < 0:
                        too_many_nested = 1

                if too_many_nested == 0:
                    if len(additional_allocations) == 1:
                        additional_allocations[0][len(additional_allocations) - 1] += 1
                    else:
                        additional_allocations[len(additional_allocations) - 1] += 1

                    remaining_part_quantities = check_totals
                    additional_allocations[0][-1] += 1

                if (remaining_part_quantities == np.zeros((num_parts, 1))).all():
                    parts_need_nested = 0

        additional_patterns = np.delete(additional_patterns, 0, axis=1)
        additional_patterns = np.delete(additional_patterns, 0, axis=1)
        additional_allocations = np.delete(additional_allocations, 0, axis=1)
        additional_allocations = np.delete(additional_allocations, 0, axis=1)
        additional_allocations = additional_allocations.T

        final_patterns = patterns.copy()
        final_allocations = int_allocation.copy()

        insertion_order = np.argsort(bridge_position_tracker)[::-1]

        # Insert additional patterns into main patterns according to bridge_position_tracker
        if len(bridge_position_tracker) > 0:
            bridge_position_tracker = np.concatenate((np.array([bridge_position_tracker]), np.array([np.arange(len(
                bridge_position_tracker))])), axis=0)
            bridge_position_tracker[0] = bridge_position_tracker[0][insertion_order]
            bridge_position_tracker[1] = bridge_position_tracker[1][insertion_order]
            for tracker_element_index, tracker_element in enumerate(bridge_position_tracker[0]):
                final_patterns = np.insert(final_patterns.T, tracker_element, additional_patterns.T[
                    bridge_position_tracker[1][tracker_element_index]].T, axis=0).T
                final_allocations = np.insert(final_allocations, tracker_element, additional_allocations[
                    bridge_position_tracker[1][tracker_element_index]], axis=0)

            # Remove bridge patterns from additional_patterns and additional_allocations
            for index in range(len(bridge_position_tracker[0])):
                additional_patterns = np.delete(additional_patterns.T, 0, axis=0).T
                additional_allocations = np.delete(additional_allocations, 0, axis=0)

        # Concatenate remaining additional patterns to main patterns if they were added
        if len(additional_allocations) != 0:
            final_patterns = np.concatenate((final_patterns, additional_patterns), axis=1)
            final_allocations = np.concatenate((final_allocations, additional_allocations))

        # Remove any patterns with final allocations of zero
        for i in range(len(final_allocations))[::-1]:
            if final_allocations[i] == 0:
                final_patterns = np.delete(final_patterns, i, 1)
                final_allocations = np.delete(final_allocations, i, 0)

        # Combine any duplicate patterns
        for i in range(np.size(final_allocations))[::-1]:
            for j in range(i)[::-1]:
                if np.all(final_patterns[:, i] == final_patterns[:, j]):
                    final_allocations[j] += final_allocations[i]
                    final_patterns = np.delete(final_patterns, i, 1)
                    final_allocations = np.delete(final_allocations, i, 0)

                    break

        nested_qtys = np.sum(np.dot(final_patterns, final_allocations))
        original_qtys = np.sum(initial_part_quantities)
        if nested_qtys != original_qtys:
            print("Nested qtys did not match original qtys!!!  Code needs to be debugged")

        # Find scrap rates for each nest and overall
        scrap_rates = 1 - np.dot(self.calcParams.part_lengths.T, final_patterns) / self.calcParams.stock_length
        overall_scrap = np.dot(scrap_rates, final_allocations) / np.sum(final_allocations)
        overall_scrap = overall_scrap[0][0]

        index_largest_drop = np.argmax(scrap_rates)
        max_drop_length = self.calcParams.stock_length - self.calcParams.right_waste
        max_drop_length -= np.dot(final_patterns.T[index_largest_drop].T, nested_lengths)[0]
        scrap_adjustment = max_drop_length / self.calcParams.stock_length / np.sum(final_allocations)
        scrap_without_drop = overall_scrap - scrap_adjustment

        actual_max_containers = int(np.max(column_sort.count_containers(final_patterns)))
        final_required_lengths = int(np.sum(final_allocations))
        final_required_lengths_minus_drop = \
            final_required_lengths - max_drop_length / self.calcParams.stock_length

        if np.size(final_patterns) >= 3:
            drop_at_end_sequence = np.arange(len(final_allocations))
            drop_at_end_sequence = np.delete(drop_at_end_sequence, index_largest_drop)
            drop_at_end_sequence = np.append(drop_at_end_sequence, index_largest_drop)
            drop_at_end_patterns = final_patterns.T[drop_at_end_sequence].T

            # Improve max containers further if possible
            if np.size(drop_at_end_patterns, 1) > 1:
                final_sequence_adjustment = column_sort.optimize_sequence(drop_at_end_patterns, 1, 1)[0]
            else:
                final_sequence_adjustment = [0]
            possible_pattern_replacement = drop_at_end_patterns.T[final_sequence_adjustment].T
            possible_pattern_replacement = column_sort.pre_process(possible_pattern_replacement)

            # TODO allow for more than max containers in bridge patterns

            final_patterns = drop_at_end_patterns.T[final_sequence_adjustment].T
            final_allocations = final_allocations[drop_at_end_sequence][final_sequence_adjustment]
            scrap_rates = scrap_rates[0][drop_at_end_sequence][final_sequence_adjustment]

            warning_messages.append(column_sort.count_containers(possible_pattern_replacement))

            actual_max_containers = int(np.max(column_sort.count_containers(possible_pattern_replacement)))
            final_required_lengths = int(np.sum(final_allocations))
            final_required_lengths_minus_drop = \
                final_required_lengths - max_drop_length / self.calcParams.stock_length

            # Sort parts (rows) by start_pattern (earliest to latest)
            column_sort.pre_process(final_patterns)
            sorted_by_start = np.argsort(column_sort.start_pattern)
            final_patterns = final_patterns[sorted_by_start]
            self.calcParams.part_quantities = self.calcParams.part_quantities[sorted_by_start]
            self.calcParams.part_names = self.calcParams.part_names[sorted_by_start]
            self.calcParams.part_lengths = self.calcParams.part_lengths[sorted_by_start]
            self.nested_lengths = nested_lengths[sorted_by_start]

        warning_messages.append(final_patterns)

        adj_matrix = self.find_adj_matrix(final_patterns)
        connected_parts_lists = []

        for i in range(num_parts):
            connected_parts_list = [i]
            for sublist_index, sublist in enumerate(connected_parts_lists):
                if i in sublist:
                    connected_parts_list = sublist
                    connected_parts_lists.pop(sublist_index)
            for element_index, element in enumerate(adj_matrix[i]):
                if element != 0:
                    for sublist_index, sublist in enumerate(connected_parts_lists.copy()):
                        if element_index in sublist:
                            connected_parts_list = self.union(connected_parts_list, sublist)
                            connected_parts_lists.pop(sublist_index)
                        else:
                            if element_index not in connected_parts_list:
                                connected_parts_list.append(element_index)
            connected_parts_lists.append(connected_parts_list)

        # Find sublist containing the parts in the pattern with the highest scrap.  Move to end.
        index_largest_drop = np.argmax(scrap_rates)
        for part_index, item in enumerate(final_patterns.T[index_largest_drop]):
            if item != 0:
                for sublist_index, sublist in enumerate(connected_parts_lists):
                    if part_index in sublist:
                        connected_parts_lists.pop(sublist_index)
                        connected_parts_lists.append(sublist)
                        break

        for sublist in connected_parts_lists:
            sublist.sort()

        reordered_connected_parts_lists = copy.deepcopy(connected_parts_lists)

        i = 0
        for sublist_index, sublist in enumerate(reordered_connected_parts_lists):
            for element_index, element in enumerate(sublist):
                reordered_connected_parts_lists[sublist_index][element_index] = i
                i += 1

        sequenced_by_networks = [item for sublist in connected_parts_lists for item in sublist]

        final_patterns = final_patterns[sequenced_by_networks]
        self.calcParams.part_quantities = self.calcParams.part_quantities[sequenced_by_networks]
        self.calcParams.part_names = self.calcParams.part_names[sequenced_by_networks]
        self.calcParams.part_lengths = self.calcParams.part_lengths[sequenced_by_networks]
        nested_lengths = nested_lengths[sequenced_by_networks]

        intermediate_pattern_sequence = []
        for sublist_index, sublist in enumerate(reordered_connected_parts_lists):
            for pattern_index, pattern in enumerate(final_patterns.T):
                for i in range(len(sublist)):
                    if pattern[sublist[i]] != 0:
                        if pattern_index not in intermediate_pattern_sequence:
                            intermediate_pattern_sequence.append(pattern_index)
                            break

        final_patterns = final_patterns.T[intermediate_pattern_sequence].T
        final_allocations = final_allocations[intermediate_pattern_sequence]

        column_sort.pre_process(final_patterns)
        split_points = np.array([0])

        adj_matrix = self.find_adj_matrix(final_patterns, num_parts)

        for i in range(num_parts - 1):
            if adj_matrix[i][i + 1] == 0:
                if np.any(adj_matrix[0: i + 1, i + 1: num_parts - 1] != 0):
                    continue
                else:
                    split_points = np.append(split_points, i + 1)

        split_points = np.append(split_points, num_parts)

        largest_in_set = np.zeros((1, len(split_points) - 1))

        for split_point_index, split_point in enumerate(split_points[:-1]):
            for i in range(split_point, split_points[split_point_index + 1]):
                if nested_lengths[i][0] > largest_in_set[0][split_point_index]:
                    largest_in_set[0][split_point_index] = nested_lengths[i][0]

        pattern_split_points = split_points.copy()

        for i in range(len(pattern_split_points) - 1):
            pattern_split_points[i] = column_sort.start_pattern[split_points[i]]

        pattern_split_points[-1] = len(final_allocations)

        order_of_chains = np.argsort(largest_in_set[0][:-1])[::-1]
        order_of_chains = np.append(order_of_chains, len(order_of_chains))

        unchanged_sequence = np.arange(len(final_allocations))
        chains = column_sort.list_of_lists(len(order_of_chains), 0)
        for i in range(len(order_of_chains)):
            chains[i] = unchanged_sequence[pattern_split_points[i]:pattern_split_points[i + 1]]

        organized_sequence = []
        for i in range(len(order_of_chains)):
            organized_sequence = np.append(organized_sequence, chains[order_of_chains[i]])

        organized_sequence = organized_sequence.astype(int)

        final_patterns = final_patterns.T[organized_sequence].T
        final_allocations = final_allocations[organized_sequence]

        # Sort parts (rows) by start_pattern (earliest to latest)
        column_sort.pre_process(final_patterns)
        sorted_by_start = np.argsort(column_sort.start_pattern)
        final_patterns = final_patterns[sorted_by_start]
        self.calcParams.part_quantities = self.calcParams.part_quantities[sorted_by_start]
        self.calcParams.part_names = self.calcParams.part_names[sorted_by_start]
        self.calcParams.part_lengths = self.calcParams.part_lengths[sorted_by_start]
        nested_lengths = nested_lengths[sorted_by_start]

        # Sort rows by shortest part to the longest part so that shorter parts end up on left
        sorted_by_length = np.argsort(nested_lengths.T[0])
        final_patterns = final_patterns[sorted_by_length]
        self.calcParams.part_lengths = self.calcParams.part_lengths[sorted_by_length]
        self.calcParams.part_names = self.calcParams.part_names[sorted_by_length]
        self.calcParams.part_quantities = self.calcParams.part_quantities[sorted_by_length]

        return final_patterns, final_allocations

    def branch_bound(
            self,
            bandwidth,
            max_parts_per_nest,
            part_quantities,
            parts_sublist,
            mode,
            patterns,
            nested_lengths,
            nestable_length,
            part_index,
            remaining_part_quantities,
            unavailable_parts,
            container_counter,
            num_parts,
            bonus_sublist,
            current_part_index=0
    ):
        # Find pi vector from patterns (Step #2)
        if mode == 0:
            patterns_inv = np.linalg.inv(patterns)
            patterns_trans_inv = np.transpose(patterns_inv)
            ones_vector = np.ones((len(patterns[0]), 1))
            pi = np.dot(patterns_trans_inv, ones_vector)
        elif mode == 1 or mode == 2:
            pi = nested_lengths / nestable_length

        # Adjust pi slightly to prioritize longer parts
        for i in range(len(pi)):
            pi[i] = pi[i] + 0.0001 * nested_lengths[i] / nestable_length

        # Calculate allocations of each pattern
        allocation = np.dot(patterns_inv, part_quantities)

        # Find value vector that can be used to prioritize the "usefulness" of nesting each part
        values = np.divide(pi, nested_lengths)

        if mode == 1:
            part_quantities[part_index] = remaining_part_quantities[part_index].copy()
            for part in unavailable_parts[1:]:
                part_quantities[part] = remaining_part_quantities[part].copy()

        values_sublist = values[parts_sublist[container_counter].astype(int)]
        pi_sublist = pi[parts_sublist[container_counter].astype(int)]
        nested_lengths_sublist = nested_lengths[parts_sublist[container_counter].astype(int)]
        part_quantities_sublist = part_quantities[parts_sublist[container_counter].astype(int)]

        # Sort from "most valuable to nest" to "least valuable to nest" so that optimum solution is reached sooner.
        index_order = (np.argsort((values * -1).transpose()))[0]
        pi_sorted = pi[index_order]
        values_sorted = values[index_order]
        nested_lengths_sorted = nested_lengths[index_order]
        part_quantities_sorted = part_quantities[index_order]

        # Sort sublist data from "most valuable to nest" to "least valuable to nest" so that optimum solution is
        # reached sooner.
        sublist_index_order = (np.argsort((values_sublist * -1).transpose()))[0]
        if mode == 1:
            sublist_index_order = np.delete(
                sublist_index_order,
                np.where(sublist_index_order == current_part_index)[0]
            )
            sublist_index_order = np.insert(sublist_index_order, 0, current_part_index)

        pi_sublist_sorted = pi_sublist[sublist_index_order]
        values_sublist_sorted = values_sublist[sublist_index_order]
        nested_lengths_sublist_sorted = nested_lengths_sublist[sublist_index_order]
        part_quantities_sublist_sorted = part_quantities_sublist[sublist_index_order]
        parts_sublist_sorted = np.zeros((num_parts - bandwidth + 1, bandwidth))

        if mode == 1 or mode == 2:
            parts_sublist_sorted = np.zeros((1, bandwidth))
        if mode == 1:
            bonus_sublist_sorted = bonus_sublist[0][sublist_index_order]
        parts_sublist_sorted[container_counter] = \
            parts_sublist[container_counter][sublist_index_order]
        # old_index_order = np.argsort(index_order)

        # Initialize branch and bound matrix, bbm, with level 1 node
        #   Row will be [level a0_1 a0_2 ... a0_n rem LP IP]
        #   bbm will consist of all nodes that may still be explored
        bbm = np.zeros((1, bandwidth + 4))
        bbm[0, 0] = 1  # level
        # Entries 1 through num_parts will remain at 0 for the first node because no parts have been nested
        bbm[0, bandwidth + 1] = nestable_length  # Calculate rem, remaining nestable length
        bbm[0, bandwidth + 2] = bbm[0, bandwidth + 1] * values_sublist_sorted[0]  # Calculate
        # value of LP, linear programming maximum value
        # bbm[0, num_parts + 3] = 0  # Value of IP, will remain at 0 since no parts are nested

        # Initialize lp_best and ip_best
        lp_best = bbm[0, bandwidth + 2]
        ip_best = 0

        # Initialize ip_best_row to keep track of best node (the one with highest IP value)
        ip_best_row = bbm[0]

        # Initialize lp_best_index at 0 since there is only one node
        lp_best_index = 0

        # Initialize loop_count
        loop_count = 0

        # Begin loop to start branching nodes, and allow for rounding error
        while lp_best > ip_best - 0.00000001:
            # Check if calculation has been canceled.
            if self.calculation_was_canceled == 1:
                # Zero out all outputs and exit function
                final_patterns = []
                final_allocations = 0
                return 0
            #     return required_lengths, allocation, patterns
            #     # return final_patterns, final_allocations

            # Extract the row to be explored
            row = bbm[lp_best_index, :]

            # Extract the level of the row to be explored
            level = int(row[0])

            # Extract the remaining length for the row to be explored
            rem = row[bandwidth + 1]

            # Extract length of part being considered at current level (level 1 corresponds to part 1 and so on)
            p_length = nested_lengths_sublist_sorted[level - 1]

            # Check how many of the part can be nested on remaining length rem
            num = math.floor(rem / p_length)

            # Reduce num if there are not enough parts available in the job to add to the nest
            part_max = int(math.floor(part_quantities_sublist_sorted[level - 1].item()))
            if part_max < num:
                num = part_max

            # Allow node to be explored by default
            # Set a variable to 1 to allow the node to be branched into sub-nodes
            branch_node = 1

            # Limit the number of parts that can be used in a pattern
            if level > max_parts_per_nest:
                # Count number of different parts in the pattern so far
                parts_in_pattern = 0
                for i in range(level):
                    if row[i + 1] != 0:
                        parts_in_pattern += 1
                # Check when only one more part can be added to the pattern
                if parts_in_pattern == max_parts_per_nest - 1:
                    # Copy the node 1 time, and iterate a0_level to num since no other parts will be nested later
                    row_copy = row.copy()  # copy row
                    row_copy[level] = num  # iterate a0_level
                    row_copy[bandwidth + 1] = rem - num * p_length  # subtract nested parts from rem
                    row_copy[bandwidth + 3] = np.dot([row_copy[1:bandwidth + 1]],
                                                     pi_sublist_sorted)  # calculate IP for new node

                    # Check if current pattern fulfills subsequent uff parts without creating more uff parts
                    #  upstream.  Give bonus IP when this occurs.
                    if mode == 1:

                        bonus, row_copy = self.check_for_bonus_condition(bonus_sublist_sorted, row_copy)

                        if bonus == 1:
                            row_copy[bandwidth + 3] += 0.08

                    # Add slight incentive to reduce number of different parts in each pattern (prevents unnecessary
                    #   mixing
                    num_parts_in_nest = 0
                    for value in row_copy[1:(bandwidth + 1)]:
                        if value:
                            num_parts_in_nest += 1
                    if num_parts_in_nest > 0:
                        row_copy[bandwidth + 3] += 1 - num_parts_in_nest / (num_parts_in_nest - 0.001)

                    if row_copy[bandwidth + 3] > ip_best:
                        ip_best = row_copy[bandwidth + 3]
                        ip_best_row = row_copy

                    # If no parts were added to the nest, keep branching.
                    if num == 0:
                        branch_node = 1
                    else:
                        branch_node = 0
                        # Remove the explored node from bbm
                        bbm = np.delete(bbm, lp_best_index, 0)
                        # Iterate original row to next level to allow for other parts to be added as final part instead
                        if level != bandwidth:
                            row[0] = level + 1
                            bbm = np.append(bbm, [row], axis=0)  # add new node to bbm

                elif parts_in_pattern == max_parts_per_nest:
                    bbm = np.delete(bbm, lp_best_index, 0)
                    branch_node = 0

            if branch_node == 1:
                # Copy the node (num + 1) times to explore it, and iterate the qty for the current part from 0 to num
                selected_range = range(num + 1)
                # Only consider the option of using all remaining parts if considering the current unfulfilled part
                if mode == 1 and level == 1:
                    selected_range = range(num, num + 1)
                for i in selected_range:
                    row_copy = row.copy()  # copy row
                    row_copy[level] = i  # iterate qty for the current part
                    row_copy[bandwidth + 1] = rem - i * p_length  # subtract nested parts from rem
                    row_copy[bandwidth + 3] = np.dot([row_copy[1:bandwidth + 1]],
                                                     pi_sublist_sorted)  # calculate IP for new node

                    # Check if current pattern fulfills subsequent uff parts without creating more uff parts
                    #  upstream.  Give bonus IP when this occurs.
                    if mode == 1:

                        bonus, row_copy = self.check_for_bonus_condition(bonus_sublist_sorted, row_copy)

                        if bonus == 1:
                            row_copy[bandwidth + 3] += 0.08

                    # Add slight incentive to reduce number of different parts in each pattern (prevents unnecessary
                    #   mixing
                    num_parts_in_nest = 0
                    for value in row_copy[1:(bandwidth + 1)]:
                        if value:
                            num_parts_in_nest += 1
                    if num_parts_in_nest > 0:
                        row_copy[bandwidth + 3] += 1 - num_parts_in_nest / (num_parts_in_nest - 0.001)

                    if row_copy[bandwidth + 3] > ip_best:
                        ip_best = row_copy[bandwidth + 3]
                        ip_best_row = row_copy
                    if level < bandwidth:
                        row_copy[bandwidth + 2] = row_copy[bandwidth + 3] + row_copy[
                            bandwidth + 1] * values_sublist_sorted[
                                                      level]  # calculate LP for new node
                        if row_copy[bandwidth + 2] > ip_best:
                            row_copy[0] = level + 1  # increment level of copy
                            bbm = np.append(bbm, [row_copy], axis=0)  # add new node to bbm

                # Remove the explored node from bbm
                bbm = np.delete(bbm, lp_best_index, 0)

            # Every 100 iterations, check for any nodes with an LP less than ip_best and remove them
            if loop_count / 100 == round(loop_count / 100):
                for i in (range(len(bbm[:, bandwidth + 2])))[::-1]:
                    if bbm[i, bandwidth + 2] < ip_best:
                        bbm = np.delete(bbm, i, 0)

            # Check to make sure bbm is not empty
            if np.size(bbm) > 0:

                # TODO Find a faster way to choose next node without cycling through all of bbm
                # Decide which node to explore next by searching for the node in bbm with highest LP
                lp_best_index = np.argmax(bbm[:, bandwidth + 2])
                lp_best = bbm[lp_best_index, bandwidth + 2]
            else:
                break

            if loop_count == 10000:
                break

            # Keep track of how many times the loop is executed
            loop_count += 1

        return ip_best, ip_best_row, index_order

    def column_gen(
            self,
            bandwidth,
            max_parts_per_nest,
            part_quantities,
            parts_sublist,
            limit_iterations,
            num_parts,
            nestable_length,
            nested_lengths,
            max_iterations,
            loop_count,
            index_order,
            parts_sublist_sorted,
            ip_best,
            ip_best_row
    ):

        # Initialize patterns matrix (single part nesting patterns)
        patterns = np.zeros((num_parts, num_parts))
        for i in range(num_parts):
            patterns[i, i] = min(self.calcParams.part_quantities[i],
                                 math.floor(nestable_length / nested_lengths[i]))

        # start_time_cg = time.time() + 1000
        # start_time_cg = time.time()

        # Reset counters
        container_counter = 0
        iteration_count = 0

        # Initialize a tracker to check for periodic cycling in ip_best
        # TODO make this less arbitrary, should be based on bandwidth too
        ip_best_history = np.array(range(num_parts * 20))

        # TODO find a way to check if minimum is global or local, try different initial conditions?

        # Execute main program loop until optimum solution is reached
        # while time.time() - start_time_cg < time_limit:
        while iteration_count < max_iterations:
            if self.calculation_was_canceled == 1:
                # Zero out all outputs and exit function
                final_patterns = []
                final_allocations = 0
                return required_lengths, allocation, patterns

            self.branch_bound(
                    bandwidth,
                max_parts_per_nest,
                part_quantities,
                parts_sublist,
                0
            )
            if iteration_count % 10 == 0:
                print(f"Number of passes during branch_bound: {loop_count}")

            # Check if calculation has been canceled.
            if self.calculation_was_canceled == 1:
                # Zero out all outputs and exit function
                final_patterns = []
                final_allocations = 0
                return required_lengths, allocation, patterns

            if ip_best >= 1:
                # Extract the best pattern from ip_best_row
                best_pattern_sorted_sublist = np.transpose([ip_best_row[1:(bandwidth + 1)]])

                # Initialize best_pattern_sorted
                best_pattern_sorted = np.zeros((num_parts, 1))

                # Add pattern values from the sublist to the main list
                for i in range(bandwidth):
                    # Find index in main list corresponding to ith index of parts_sublist_sorted
                    corr_index = \
                        np.where(index_order == parts_sublist_sorted[container_counter][i])[0].item()
                    best_pattern_sorted[corr_index] = best_pattern_sorted_sublist[i]

                # Reorder best_pattern vector
                old_index_order = np.argsort(index_order)
                best_pattern = best_pattern_sorted[old_index_order]

                # Determine which pattern to replace in patterns
                # Solve for p_bar_j (proportion of each existing pattern that 1 instance of best_pattern can replace)
                p_bar_j = np.dot(patterns_inv, best_pattern)

                # Initialize theta_limits
                theta_limits = np.zeros((num_parts, 1))

                # Calculate required_lengths before adding new pattern to patterns
                required_lengths = np.sum(np.dot(patterns_inv, part_quantities))

                # Find limiting pattern when replacing current nests with the max number of instances of best_pattern
                for i in range(num_parts):
                    # Solve for the reciprocal of the limits on Theta for each pattern
                    theta_limits[i] = p_bar_j[i] / (allocation[i] + 0.00000001)

                # Replace the pattern with the largest value of theta_limits
                index_to_replace = np.argmax(theta_limits)
                patterns[:, index_to_replace] = best_pattern.transpose()[0]

                # Fixes issue with allocation not matching new version of pattern TODO figure out why
                patterns_inv = np.linalg.inv(patterns)
                allocation = np.dot(patterns_inv, part_quantities)

            # Update ip_best_history
            ip_best_history = np.append(ip_best_history[1:], ip_best)
            iteration_count += 1

            # If repetitions are found in ip_best_history, then exit the column generation function with results.
            if self.find_cycling(ip_best_history):
                # print(time.time() - start_time_cg)
                print(f"cycling found after {iteration_count} iterations")
                # decrement max_iterations to make cycling less likely
                if limit_iterations == 1:
                    if max_iterations == 1000000:  # TODO recode since this condition is only met on first loop
                        max_iterations = iteration_count
                    else:
                        max_iterations = self.adjust_max_iterations(0.95, max_iterations)

                return required_lengths, allocation, patterns, max_iterations

            # Reset container_counter when it reaches num_parts - bandwidth + 1
            container_counter += 1
            if container_counter == num_parts - bandwidth + 1:
                container_counter = 0

        print(f"reached maximum iterations of {max_iterations}")
        # increment max_iterations to make cycling more likely
        if limit_iterations == 1:
            max_iterations = self.adjust_max_iterations(1.01, max_iterations)

        return required_lengths, allocation, patterns, loop_count, index_order, ip_best, ip_best_row

    @staticmethod
    def check_for_bonus_condition(bonus_sublist_sorted, row_copy):
        bonus = 1
        for check_index in range(len(bonus_sublist_sorted)):
            if bonus_sublist_sorted[check_index] == 1:  # Indicates parts that fulfill uff parts
                if row_copy[check_index + 1]:
                    continue
                else:
                    bonus = 0
                    break
            elif bonus_sublist_sorted[check_index] == -1:  # Prevents creation of more uff parts
                if not row_copy[check_index + 1]:
                    continue
                else:
                    bonus = 0
                    break

        return bonus, row_copy

    @staticmethod
    def adjust_max_iterations(self, adjustment_factor, max_iterations):
        max_iterations *= adjustment_factor
        if adjustment_factor >= 1:
            max_iterations += 1
        elif adjustment_factor <= 1:
            max_iterations -= 1

        return max_iterations

    @staticmethod
    def find_adj_matrix(patterns, num_parts):
        adj_matrix_ = np.zeros((num_parts, num_parts))
        for row_index, row in enumerate(patterns):
            for term_index, term in enumerate(row):
                if term:
                    for item_index, item in enumerate(patterns.T[term_index]):
                        if item:
                            adj_matrix_[row_index][item_index] = 1
        for i in range(num_parts):
            adj_matrix_[i][i] = 0

        return adj_matrix_

    @staticmethod
    def union(a, b):
        for item in a:
            if item not in b:
                b.append(item)
        return b

    @staticmethod
    def find_cycling(original_list):
        # Reverse the order of the list to make it easier to work with since repetitions would happen at end
        reversed_list = original_list[::-1]

        # Create generator to find next item in list
        index_generator = (i for i, e in enumerate(reversed_list) if e == reversed_list[0])

        # Check for first index before entering loop to prevent false positive
        first_index = next(index_generator)

        # Check for periodic nature in loop until function exits with return
        while 1:
            # Find next element that matches the first element
            try:
                next_match = next(index_generator)
            except StopIteration:
                return False

            if next_match < -1:
                pass
            else:
                if np.all(reversed_list[0:next_match] == reversed_list[next_match:(2 * next_match)]) and \
                        next_match > 10:

                    return True
