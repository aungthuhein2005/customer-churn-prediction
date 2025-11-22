import { createHashRouter } from "react-router";
import { Layout } from "../components/Layout";
import { UploadScreen } from "../components/UploadScreen";
import { ColumnMapping } from "../components/ColumnMapping";
import { PreprocessingOptions } from "../components/PreprocessingOptions";
import { ResultsDashboard } from "../components/ResultsDashboard";

export const router = createHashRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: UploadScreen },
      { path: "mapping", Component: ColumnMapping },
      { path: "preprocessing", Component: PreprocessingOptions },
      { path: "results", Component: ResultsDashboard },
    ],
  },
]);