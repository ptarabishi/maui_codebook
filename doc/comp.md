# Component Specifications

### Data Formatter
   1. Time alignment
      - Input: TTL (path, csv)
      - Output: 
   2. Motion correction
      - Input: Imaging data (path, nii)
      - Output: Motion-corrected data (nii)
   3. df/F Cacluator - should come after clusterer
      - Input: Motion-corrected data (nii)
      - Output: df/F for each cluster (h5)
      - Test: Check for correct number of clusters
   4. ROI Clusterer
      - Input: Motion-corrected data (nii)
      - Output: Pixel data with ROI ids (h5)
      - Tests: check for correct number of pixels (should be same shape as original nifti)
   5. Metadata Saver
      - Input: Imaging data (path, nii)
      - Output: Brain dimensions (h5)
### Visualization Manager
   - Inputs:
     - Slice, ROI, df/F (dataframe)
   - Outputs: 
     - df/F over time
     - spatial organization of ROIs
   - Tests: 
     - Pattern: check for all the correct information and formatting