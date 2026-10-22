export interface LoanInput {
  loan_amnt: number;
  annual_inc: number;
  dti: number;
  fico_range_low: number;
  fico_range_high: number;
  delinq_2yrs: number;
  inq_last_6mths: number;
  open_acc: number;
  pub_rec: number;
  revol_bal: number;
  revol_util: number;
  total_acc: number;
  term: string;
  emp_length: string;
  home_ownership: string;
  verification_status: string;
  purpose: string;
}

export interface ShapFactor {
  feature: string;
  impact: number;
}

export interface ScoreResponse {
  // Fields returned by the FastAPI prediction endpoint
  raw_default_probability: number;
  default_probability: number;
  risk_score: number;
  score_min: number;
  score_max: number;
  risk_band: string;
  loan_amount: number;
  expected_loss: number;
  lgd_assumption: number;
  score_type: string;

  // Optional fields for dashboard features
  explanation?: string;
  top_factors?: ShapFactor[];
  disclaimer?: string;
}