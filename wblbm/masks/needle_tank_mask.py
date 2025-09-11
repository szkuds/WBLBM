import numpy as np

def needle_tank_mask(Ny, Nx, theta, Rd):
    """
    A function that generates a mask of a T-junction system. T if node is a fluid element, F if node is a wall.

    parameters:

        Ny (int): size of y-dimension of the output matrix
        Nx (int): size of x-dimension of the output matrix
        theta (float): angle between the two needles (in radians)
        Rd (float): droplet radius

    returns:

        mask (NDarray): a matrix consisting of the mask corresponding that resembles a needle in a tank
        Rx (int): x-coordinate of the centre of the droplet
        Ry (int): y-coordinate of the centre of the droplet
    """

    def line(x, a, b):

        return a * x + b

    mask = np.ones((Ny, Nx // 2 - 1))

    alpha = 90 - np.degrees(theta / 2)

    needle_thickness = 1.5
    needle_offset = 2.75


    Mx, My = Nx // 2 - 1, Ny // 2 - 1
    Rx, Ry = Mx - 1.5 * Rd * np.cos(theta / 2), My + 1.5 * Rd * np.sin(theta / 2)
    Bx, By = Mx - needle_offset * Rd * np.cos(theta / 2), My + needle_offset * Rd * np.sin(theta / 2)
    Bx_low, By_low = Bx - needle_thickness * Rd * np.cos(np.radians(alpha)), By - needle_thickness * Rd * np.sin(np.radians(alpha))
    Bx_up, By_up = Bx + needle_thickness * Rd * np.cos(np.radians(alpha)), By + needle_thickness * Rd * np.sin(np.radians(alpha))
    Cx_low, Cy_low = 0, By_low + Bx_low * np.tan(np.radians(90 - alpha))
    Cx_up, Cy_up = 0, By_up + Bx_up * np.tan(np.radians(90 - alpha))

    aBC_up, bBC_up = (By_up - Cy_up) / (Bx_up - Cx_up), Cy_up
    aBC_low, bBC_low = (By_low - Cy_low) / (Bx_low - Cx_low), Cy_low

    if Bx_up - Bx_low != 0:
        aBB, bBB = (By_up - By_low) / (Bx_up - Bx_low), By - Bx * np.tan(np.radians(alpha))

    for y in range(Ny):
        for x in range(Nx // 2):

            # topline
            if int(line(x, aBC_up, bBC_up)) == y and x <= Bx_up:
                mask[y, x] = 0
                mask[y + 1, x] = 0

            # botline
            if int(line(x, aBC_low, bBC_low)) == y and x <= Bx_low:
                mask[y, x] = 0
                mask[y - 1, x] = 0

    # Mirror the mask horizontally to create the full domain
    # Add back the first column (if needed for alignment)
    mask0 = np.copy(mask[:, 0]).reshape((Ny, 1))
    left_mask = np.concatenate((mask0, mask), axis=1)

    # Mirror the left half to create the right half
    right_mask = np.fliplr(left_mask)

    # Concatenate left and right halves to form the full tank
    full_mask = np.concatenate((left_mask, right_mask), axis=1)

    return full_mask, Rx, Ry

    return mask, Rx, Ry