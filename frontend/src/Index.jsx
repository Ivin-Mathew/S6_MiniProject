import React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import App from "./App";
import Preview from "./Preview";

ReactDOM.render(
  <Router>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/preview" element={<Preview />} />
    </Routes>
  </Router>,
  document.getElementById("root")
);


