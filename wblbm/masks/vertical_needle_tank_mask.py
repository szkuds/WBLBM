import numpy as np

def vertical_needle_tank_mask(Ny, Nx, Rd=1):
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



    mask = np.ones((Ny, Nx // 2 - 1))



    needle_thickness = Nx//20
    needle_length = Ny//2
    needle_offset_from_center = Nx//10



    #Mx, My = Nx // 2 - 1, Ny // 2 - 1

    for y in range(Ny):
        for x in range(Nx // 2):

            # topline
            if ((x==Nx//2-needle_offset_from_center-needle_thickness//2
                or x==Nx//2-needle_offset_from_center+needle_thickness//2)
                    and y <needle_length):
                mask[y, x] = 0.1

            if (x>Nx//2-needle_offset_from_center-needle_thickness//2
                and x<Nx//2-needle_offset_from_center+needle_thickness//2
                and y <needle_length):
                mask[y,x] = 0.1


    Rx = Nx//2-needle_offset_from_center
    Ry = needle_length + np.sqrt(Rd**2-(needle_thickness/2)**2)
    assert Rd > (needle_thickness/2), "Bubble radius too small"

    # Mirror the mask horizontally to create the full domain
    # Add back the first column (if needed for alignment)
    mask0 = np.copy(mask[:, 0]).reshape((Ny, 1))
    left_mask = np.concatenate((mask0, mask), axis=1)

    # Mirror the left half to create the right half
    right_mask = np.fliplr(left_mask)

    # Concatenate left and right halves to form the full tank
    full_mask = np.concatenate((left_mask, right_mask), axis=1)

    if full_mask.shape[1] < Nx:
        pad = np.ones((Ny, 1))  # or zeros if you want a wall
        full_mask = np.concatenate((full_mask, pad), axis=1)

    return full_mask, Rx, Ry

    return mask, Rx, Ry