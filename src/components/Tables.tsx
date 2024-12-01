import { Progress } from 'antd';
import type { ProgressProps } from 'antd';

const conicColors: ProgressProps['strokeColor'] = {
  '0%': '#87d068',
  '50%': '#ffe58f',
  '100%': '#ffccc7',
};

interface props {
  param: number | string;
  type: string;
}

export default function Tables({param, type}: props) {
  let container;

  if (type === 'Churn ROC AUC' || type === 'Bad loans' && typeof param === 'number') {
    container = <Progress type="dashboard" percent={Number(param)} strokeColor={conicColors} />
  } else {  // if (type === 'Total Profit') {
    container = <div className="number-value">{param}</div>
  }

  return <>
  <div className="metric">
    <p>{type}:</p>
    {container}
  </div>
  </>
}