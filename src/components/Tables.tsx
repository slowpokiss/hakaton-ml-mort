import { Progress } from 'antd';
import type { ProgressProps } from 'antd';

const conicColors: ProgressProps['strokeColor'] = {
  '0%': '#87d068',
  '50%': '#ffe58f',
  '100%': '#ffccc7',
};

interface props {
  param: number | string| any;
  type: string;
}

export default function Tables({param, type}: props) {
  let container;

  if (type === 'churn_auc' || type === 'bad_loans' && typeof param === 'number') {
    container = <Progress type="dashboard" percent={Number(param)} strokeColor={conicColors} />
    if (type === 'churn_auc') {
      container = <Progress type="dashboard" percent={Number(param * 100)} strokeColor={conicColors} />
    }
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