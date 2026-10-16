export interface LoanInput {
  loan_amnt: number; annual_inc: number; dti: number; delinq_2yrs: number;
  open_acc: number; pub_rec: number; revol_bal: number; revol_util: number;
  total_acc: number; term: string; emp_length: string; home_ownership: string;
  verification_status: string; purpose: string;
}
export interface ShapFactor { feature: string; impact: number; }
export interface ScoreResponse {
  credit_risk_score: number; score_scale: string; predicted_default_probability: number;
  risk_band: string; score_type?: string; top_factors: ShapFactor[]; explanation?: string;
  expected_loss?: number; disclaimer?: string;
}
