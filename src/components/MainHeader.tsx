import Tables from "./Tables";
import UploadComponents from "./UploadComponents";
import { useSelector } from "react-redux";
import { initialStateInterface } from "../redux/mainSlice";
import { useDispatch } from "react-redux";
import { setCurrMetrics } from "../redux/mainSlice";

export default function MainHeader() {
  const currMetrics = useSelector((state: {main: initialStateInterface}) => state.main.currMetrics);
  const dispatch = useDispatch()

  return  <>
  <header>
    <h1 className="header">Без шансов</h1>
  </header>
  <main>
    <UploadComponents />
    <div className="btn" onClick={() => dispatch(setCurrMetrics())}>Показать</div>
    <div className="tables" style={{ display: `${currMetrics !== false ? 'flex' : 'none' } ` }}>
      <Tables param={0.962 * 100} type={'Churn ROC AUC'} />
      <Tables param={'-0.114'} type={'Price NMSLE'} />
      <Tables param={2.2} type={'Bad loans'} />
      <Tables param={19111.0} type={'Issue Amount'} />
      <Tables param={6791.0} type={'Total Profit'} />
    </div>
  </main>
  </>;
}