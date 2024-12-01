import { createSlice } from "@reduxjs/toolkit";

export interface initialStateInterface {
  currMetrics: any;
}

const initialState: initialStateInterface = {
  currMetrics: [],
};

export const mainSlice = createSlice({
  name: "mainSlice",
  initialState,
  reducers: {
    setCurrMetrics(state, action) {
      state.currMetrics = action.payload;
    },
  },
});

export const { setCurrMetrics } = mainSlice.actions;
export default mainSlice.reducer;
