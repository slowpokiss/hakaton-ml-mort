import { createSlice } from "@reduxjs/toolkit";

export interface initialStateInterface {
  currMetrics: any;
}

export const mainSlice = createSlice({
  name: "mainSlice",
  initialState: {
    currMetrics: ''
  } satisfies initialStateInterface as initialStateInterface,
  reducers: {
    setCurrMetrics(state) {
      console.log(state)
      state.currMetrics = !state.currMetrics;
    },
  }
});

export const { setCurrMetrics } = mainSlice.actions;
export default mainSlice.reducer;
