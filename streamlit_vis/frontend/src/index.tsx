// streamlit_vis/frontend/src/index.tsx
import React from "react";
import ReactDOM from "react-dom";
import VisNetwork from "./VisNetwork";

// @ts-ignore
const streamlitDoc = window.document;

ReactDOM.render(
  <React.StrictMode>
    <VisNetwork />
  </React.StrictMode>,
  streamlitDoc.getElementById("root")
);