import { RouterProvider } from 'react-router';
import { Provider } from 'react-redux';
import { store } from './store/store';
import { router } from './utils/routes';

export default function App() {
  return (
    <Provider store={store}>
      <RouterProvider router={router} />
    </Provider>
  );
}
