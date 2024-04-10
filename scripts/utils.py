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