import Tables from "./Tables";
import UploadComponents from "./UploadComponents";
import { useSelector } from "react-redux";
import { initialStateInterface } from "../redux/mainSlice";

export default function MainHeader() {
  const currMetrics = useSelector((state: {main: initialStateInterface}) => state.main.currMetrics);
  
  return  <>
  <header>
    <h1 className="header">Без шансов</h1>
  </header>
  <main>
    <UploadComponents />
    <div className="tables" style={{ display: `${currMetrics !== false ? 'flex' : 'none' } ` }}>
      {
        currMetrics.map((el: any) => {
          return <Tables param={el.test} type={el.index} />
        })
      }

{/* {
    "index": "issue_amount",
    "test": 21990,
    "desc": "Итоговая выданная сумма (25 000 максимум), млн руб."
  }, */}

      {/* <Tables param={0.962 * 100} type={'Churn ROC AUC'} />
      <Tables param={'-0.114'} type={'Price NMSLE'} />
      <Tables param={2.2} type={'Bad loans'} />
      <Tables param={19111.0} type={'Issue Amount'} />
      <Tables param={6791.0} type={'Total Profit'} /> */}
    </div>
  </main>
  </>;
}