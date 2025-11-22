import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UploadState {
  fileName: string;
  isUploading: boolean;
  error: string | null;
}

const initialState: UploadState = {
  fileName: '',
  isUploading: false,
  error: null,
};

const uploadSlice = createSlice({
  name: 'upload',
  initialState,
  reducers: {
    setFileName: (state, action: PayloadAction<string>) => {
      state.fileName = action.payload;
    },
    setUploading: (state, action: PayloadAction<boolean>) => {
      state.isUploading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    resetUpload: (state) => {
      state.fileName = '';
      state.isUploading = false;
      state.error = null;
    },
  },
});

export const { setFileName, setUploading, setError, resetUpload } = uploadSlice.actions;
export default uploadSlice.reducer;

