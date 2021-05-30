# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

#import numpy as np
#cimport numpy as np

## TODO: Allocate memory internally. See: https://stackoverflow.com/questions/18462785/what-is-the-recommended-way-of-allocating-memory-for-a-typed-memory-view

def adjacency_matrix( image, num_regions ):
    '''
    Given:
        image: A 2D image of integer labels in the range [0,num_regions].
        num_regions: The number of regions in `image`.
    Returns:
        A: The adjacency matrix such that A[i,j] is 1 if region i is
           connected to region j and 0 otherwise.
    '''
    
    import numpy as np
    A = np.zeros( ( num_regions, num_regions ), dtype = int )
    adjacency_matrix_internal( image, A )
    return A

cpdef long[:,:] adjacency_matrix_internal( long[:,:] image, long[:,:] A ) nogil:
    '''
    Given:
        image: A 2D image of integer labels in the range [0,num_regions].
    Returns:
        A: The adjacency matrix such that A[i,j] is 1 if region i is
           connected to region j and 0 otherwise.
    
    Note: `A` is an output parameter. Allocate space and pass it in.
    '''
    
    # A = np.zeros( ( num_regions, num_regions ), dtype = int )
    A[:] = 0
    
    cdef long nrow = image.shape[0]
    cdef long ncol = image.shape[1]
    cdef long i,j,region0,region1
    
    ## Sweep with left-right neighbors. Skip the right-most column.
    for i in range(nrow):
        for j in range(ncol-1):
            region0 = image[i,j]
            region1 = image[i,j+1]
            A[region0,region1] = 1
            A[region1,region0] = 1
    
    ## Sweep with top-bottom neighbors. Skip the bottom-most row.
    for i in range(nrow-1):
        for j in range(ncol):
            region0 = image[i,j]
            region1 = image[i+1,j]
            A[region0,region1] = 1
            A[region1,region0] = 1
    
    ## Sweep with top-left-to-bottom-right neighbors. Skip the bottom row and right column.
    for i in range(nrow-1):
        for j in range(ncol-1):
            region0 = image[i,j]
            region1 = image[i+1,j+1]
            A[region0,region1] = 1
            A[region1,region0] = 1
    
    ## Sweep with top-right-to-bottom-left neighbors. Skip the bottom row and left column.
    for i in range(nrow-1):
        for j in range(1,ncol):
            region0 = image[i,j]
            region1 = image[i+1,j-1]
            A[region0,region1] = 1
            A[region1,region0] = 1
    
    ## region will not connect to itself
    for i in range(len(A)):
        A[i, i] = 0

    return A

def region_sizes( image, num_regions ):
    '''
    Given:
        image: A 2D image of integer labels in the range [0,num_regions].
        num_regions: The number of regions in `image`.
    Returns:
        sizes: An array of length `num_regions`. Each element stores the
                      number of pixels with the corresponding region number.
                      That is, region i has `region_sizes[i]` pixels.
    '''
    
    import numpy as np
    sizes = np.zeros( num_regions, dtype = int )
    region_sizes_internal( image, sizes )
    return sizes

cpdef long[:] region_sizes_internal( long[:,:] image, long[:] sizes ) nogil:
    '''
    Given:
        image: A 2D image of integer labels in the range [0,num_regions].
    Returns:
        sizes: An array of length `num_regions`. Each element stores the
                      number of pixels with the corresponding region number.
                      That is, region i has `region_sizes[i]` pixels.
    
    Note: `sizes` is an output parameter. Allocate space and pass it in.
    '''
    
    # sizes = np.zeros( num_regions, dtype = int )
    sizes[:] = 0
    
    cdef long nrow = image.shape[0]
    cdef long ncol = image.shape[1]
    cdef long i,j,region
    
    ## Sweep with left-right neighbors. Skip the right-most column.
    for i in range(nrow):
        for j in range(ncol):
            region = image[i,j]
            sizes[region] += 1
    
    return sizes

def remap_labels( image, remaps ):
    '''
    Given:
        image: A 2D image of integer labels in the range [0,len(remaps)].
        remaps: A 1D array of integer remappings that occurred.
    Returns:
        image_remapped: An array the same size as `image` except labels have been
                        remapped according to `remaps`.
    '''
    
    import numpy as np
    image_remapped = image.copy().astype( int )
    remaps = remaps.astype( int )
    remap_labels_internal( image_remapped, remaps )
    return image_remapped

cpdef long remap_labels_internal( long[:,:] image, long[:] remaps ) nogil:
    '''
    Given:
        image: A 2D image of integer labels in the range [0,len(remaps)].
        remaps: A 1D array of integer remappings that occurred.
    Modifies in-place:
        image: `image` with labels remapped according to `remaps`.
    '''
    
    cdef long nrow = image.shape[0]
    cdef long ncol = image.shape[1]
    cdef long i,j,region
    
    ## Sweep with left-right neighbors. Skip the right-most column.
    for i in range(nrow):
        for j in range(ncol):
            region = image[i,j]
            while remaps[region] != region:
                image[i,j] = remaps[region]
                region = remaps[region]
    
    return i
