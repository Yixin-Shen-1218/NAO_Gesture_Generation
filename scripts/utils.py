def clamp_matrix(processed_matrix):
    """
    Rounds a value to -1 or 1 if it's not within the range [-1, 1].
    """
    # Find elements outside the range [-1, 1]
    mask_lower = processed_matrix < -1
    mask_upper = processed_matrix > 1

    # Clamp elements to the range [-1, 1]
    processed_matrix[mask_lower] = -1
    processed_matrix[mask_upper] = 1

    # Round elements to the nearest integer
    # processed_matrix = np.round(processed_matrix)

    return processed_matrix


def normalize(numbers):
    print numbers
    # Find the minimum and maximum values in the list
    min_val = min(numbers)
    max_val = max(numbers)

    # Normalize each number in the list
    normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]

    print("Original numbers:", numbers)
    print("Normalized numbers:", normalized_numbers)
    return normalized_numbers


def scale_to_range(float_list, target_min=0, target_max=10):
    """
    Maps a list of floats to the range [target_min, target_max].

    Parameters:
    float_list (list): A list of float values.
    target_min (float): The minimum value of the target range.
    target_max (float): The maximum value of the target range.

    Returns:
    list: A new list with the float values mapped to the target range.
    """
    # Find the minimum and maximum values in the input list
    input_min = min(float_list)
    input_max = max(float_list)

    # Map the input values to the target range
    scaled_list = [(x - input_min) / (input_max - input_min) * (target_max - target_min) + target_min for x in
                   float_list]

    return scaled_list


if __name__ == '__main__':
    # Example usage:
    numbers = [2.5, 5.0, 7.5, 10.0]
    normalized_numbers = normalize(numbers)
    print("Original numbers:", numbers)
    print("Normalized numbers:", normalized_numbers)

    float_list = [-2.5, 0.0, 5.0, 7.5, 10.0]

    # Scale the list to the range [0, 10]
    scaled_list = scale_to_range(float_list)
    print(scaled_list)