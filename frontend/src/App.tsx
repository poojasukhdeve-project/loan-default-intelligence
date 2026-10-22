import { useState, type FormEvent, type ReactNode } from 'react';
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Wallet,
  ChartNoAxesCombined,
  LoaderCircle,
} from 'lucide-react';
import { requestScore } from './api';
import type { LoanInput, ScoreResponse } from './types';

const initial: LoanInput = {
  loan_amnt: 12000,
  annual_inc: 72000,
  dti: 18.5,
  // Overwritten at submit time from the bureau scores (see submitApplication)
  fico_range_low: 680,
  fico_range_high: 684,
  delinq_2yrs: 0,
  inq_last_6mths: 1,
  open_acc: 8,
  pub_rec: 0,
  revol_bal: 8500,
  revol_util: 32,
  total_acc: 18,
  term: ' 36 months',
  emp_length: '5 years',
  home_ownership: 'MORTGAGE',
  verification_status: 'Verified',
  purpose: 'debt_consolidation',
};

const initialBureau = [720, 705, 680];
const initialStability = [-500, -3000, -4000, 0.03];

const names: Record<string, string> = {
  loan_amnt: 'Loan amount',
  annual_inc: 'Annual income',
  dti: 'Debt-to-income ratio',
  fico_score: 'FICO score',
  delinq_2yrs: 'Recent delinquencies',
  inq_last_6mths: 'Recent credit inquiries',
  open_acc: 'Open credit accounts',
  pub_rec: 'Public records',
  revol_bal: 'Revolving balance',
  revol_util: 'Revolving utilization',
  total_acc: 'Total credit accounts',
  term: 'Loan term',
  emp_length: 'Employment length',
  home_ownership: 'Home ownership',
  verification_status: 'Income verification',
  purpose: 'Loan purpose',
};

const steps = [
  { title: 'Credit Bureau Scores', short: 'Credit Scores' },
  { title: 'Applicant Profile', short: 'Applicant' },
  { title: 'Loan Details', short: 'Loan Details' },
  { title: 'Stability Indicators', short: 'Stability' },
];

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

function Num({
  label,
  value,
  onChange,
  hint,
  min = 0,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <Field label={label} hint={hint}>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        required
      />
    </Field>
  );
}

function Sel({
  label,
  value,
  options,
  onChange,
  hint,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (s: string) => void;
  hint?: string;
}) {
  return (
    <Field label={label} hint={hint}>
      <span className="select-wrap">
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((option) => (
            <option key={option} value={option}>
              {option.trim()}
            </option>
          ))}
        </select>
        <ChevronDown size={16} />
      </span>
    </Field>
  );
}

export default function App() {
  const [form, setForm] = useState<LoanInput>(initial);
  const [step, setStep] = useState(0);
  const [bureau, setBureau] = useState<number[]>(initialBureau);
  const [age, setAge] = useState(35);
  const [occupation, setOccupation] = useState('Laborers');
  const [incomeSource, setIncomeSource] = useState('Working');
  const [employer, setEmployer] = useState('Business Entity Type 3');
  const [education, setEducation] = useState('Higher education');
  const [marital, setMarital] = useState('Married');
  const [emi, setEmi] = useState(400);
  const [stability, setStability] = useState<number[]>(initialStability);
  const [region, setRegion] = useState('1');
  const [currency, setCurrency] = useState('USD');
  const [includeExplanation, setIncludeExplanation] = useState(true);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const updateNumber = (key: keyof LoanInput, value: number) => {
    setForm((old) => ({ ...old, [key]: value }));
  };

  const updateText = (key: keyof LoanInput, value: string) => {
    setForm((old) => ({ ...old, [key]: value }));
  };

  // ---- Score display values (range and band come from the backend) ----
  const score = result?.risk_score ?? 0;
  const scoreMin = result?.score_min ?? 300;
  const scoreMax = result?.score_max ?? 850;
  const scorePct = Math.max(
    0,
    Math.min(100, ((score - scoreMin) / (scoreMax - scoreMin)) * 100),
  );

  const probability = result?.default_probability ?? 0;
  const riskBand = result?.risk_band ?? 'Awaiting prediction';

  const riskClass =
    score >= 720
      ? 'low'
      : score >= 620
        ? 'medium-low'
        : score >= 520
          ? 'medium'
          : 'high';

  const annualPaymentRatio =
    form.annual_inc > 0 ? ((emi * 12) / form.annual_inc) * 100 : 0;

  const expectedLoss =
    result?.expected_loss ?? probability * form.loan_amnt * 0.45;

  const money = (value: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      maximumFractionDigits: 0,
    }).format(value);

  async function submitApplication() {
    setLoading(true);
    setError('');

    // The average of the three bureau scores feeds the model's FICO range.
    const avg = Math.round(bureau.reduce((a, b) => a + b, 0) / bureau.length);
    const ficoLow = Math.max(300, Math.min(846, avg));
    const payload: LoanInput = {
      ...form,
      fico_range_low: ficoLow,
      fico_range_high: ficoLow + 4,
    };

    try {
      const prediction = await requestScore(payload, form.loan_amnt, 0.45);
      setResult(prediction);
      setStep(4);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not connect to the prediction API. Check that your backend is running.',
      );
    } finally {
      setLoading(false);
    }
  }

  function startNewApplication() {
    setForm(initial);
    setBureau(initialBureau);
    setStability(initialStability);
    setEmi(400);
    setStep(0);
    setResult(null);
    setError('');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function nextStep(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');

    if (step < 3) {
      setStep((current) => current + 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      void submitApplication();
    }
  }

  const factors = result?.top_factors ?? [];
  const topFactors = factors.slice(0, 5);
  // Scale bars against the largest factor so they are visually comparable
  const maxImpact = Math.max(
    ...topFactors.map((f) => Math.abs(f.impact)),
    1e-9,
  );

  return (
    <main className="app">
      <header className="topbar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            startNewApplication();
          }}
        >
          <span className="brand-icon">
            <ChartNoAxesCombined size={22} />
          </span>
          <span>
            Loan Default <strong>Intelligence</strong>
          </span>
        </a>
        <nav className="topnav">
          <a href="#application">Scorecard</a>
          <a href="#results">Risk Analysis</a>
          <span className="status-pill">
            <span /> Research Demo
          </span>
        </nav>
      </header>

      <section className="page-intro">
        <div className="eyebrow">
          <Sparkles size={15} /> AI-POWERED CREDIT DECISIONING
        </div>
        <h1>Credit Risk Scorecard</h1>
        <p>
          Assess loan default risk with a model-powered score and transparent
          risk metrics.
        </p>
      </section>

      {step < 4 && (
        <section className="wizard-card" id="application">
          <div className="wizard-heading">
            <div>
              <span className="overline">LOAN APPLICATION</span>
              <h2>{steps[step].title}</h2>
              <p>Step {step + 1} of 4 · Complete the fields below to continue.</p>
            </div>
            <div className="step-count">
              {String(step + 1).padStart(2, '0')} <span>/ 04</span>
            </div>
          </div>

          <div className="progress-track" aria-label={`Step ${step + 1} of 4`}>
            {steps.map((item, index) => (
              <button
                type="button"
                key={item.title}
                className={`progress-step ${index === step ? 'active' : ''} ${index < step ? 'completed' : ''}`}
                onClick={() => index < step && setStep(index)}
                disabled={index > step}
                aria-current={index === step ? 'step' : undefined}
              >
                <span className="progress-number">
                  {index < step ? <Check size={16} /> : index + 1}
                </span>
                <span className="progress-label">{item.short}</span>
              </button>
            ))}
          </div>

          <form onSubmit={nextStep}>
            {step === 0 && (
              <div className="wizard-section">
                <div className="section-heading">
                  <span className="section-icon">
                    <ShieldCheck size={19} />
                  </span>
                  <div>
                    <h3>Credit bureau scores</h3>
                    <p>
                      The average of these scores is sent to the model as the
                      FICO range.
                    </p>
                  </div>
                </div>
                <div className="fields-grid two">
                  {bureau.map((value, index) => (
                    <Num
                      key={index}
                      label={`Bureau Score ${index + 1}`}
                      value={value}
                      min={300}
                      max={850}
                      onChange={(n) =>
                        setBureau((old) =>
                          old.map((v, i) => (i === index ? n : v)),
                        )
                      }
                      hint={
                        ['Primary', 'Secondary', 'Tertiary'][index] +
                        ' bureau score · used in model'
                      }
                    />
                  ))}
                </div>
                <div className="info-note">
                  <AlertCircle size={17} />
                  The average bureau score is sent to the model as the FICO
                  range. Valid range is 300 to 850.
                </div>
              </div>
            )}

            {step === 1 && (
              <div className="wizard-section">
                <div className="section-heading">
                  <span className="section-icon">
                    <Wallet size={19} />
                  </span>
                  <div>
                    <h3>Applicant profile</h3>
                    <p>
                      Enter employment and background details. Only supported
                      model fields are submitted.
                    </p>
                  </div>
                </div>
                <div className="fields-grid two">
                  <Num
                    label="Age (years)"
                    value={age}
                    min={18}
                    max={100}
                    onChange={setAge}
                    hint="UI reference only; not sent to model"
                  />
                  <Sel
                    label="Occupation"
                    value={occupation}
                    onChange={setOccupation}
                    options={[
                      'Laborers',
                      'Accountants',
                      'Managers',
                      'Drivers',
                      'Sales staff',
                      'IT / Technology',
                      'Self-employed',
                      'Other',
                    ]}
                    hint="UI reference only"
                  />
                  <Sel
                    label="Income Source"
                    value={incomeSource}
                    onChange={setIncomeSource}
                    options={[
                      'Working',
                      'Commercial associate',
                      'Pensioner',
                      'State servant',
                      'Self-employed',
                      'Other',
                    ]}
                    hint="UI reference only"
                  />
                  <Num
                    label="Years at Current Job"
                    value={parseInt(form.emp_length, 10) || 0}
                    min={0}
                    max={50}
                    onChange={(n) =>
                      updateText('emp_length', n === 0 ? '< 1 year' : `${n} years`)
                    }
                    hint="Mapped to employment length"
                  />
                  <Sel
                    label="Employer Type"
                    value={employer}
                    onChange={setEmployer}
                    options={[
                      'Business Entity Type 1',
                      'Business Entity Type 2',
                      'Business Entity Type 3',
                      'Government',
                      'Private company',
                      'Self-employed',
                      'Other',
                    ]}
                    hint="UI reference only"
                  />
                  <Sel
                    label="Education Level"
                    value={education}
                    onChange={setEducation}
                    options={[
                      'Higher education',
                      'Secondary education',
                      'Incomplete higher',
                      'Lower secondary',
                      'Academic degree',
                      'Other',
                    ]}
                    hint="UI reference only"
                  />
                  <Sel
                    label="Marital Status"
                    value={marital}
                    onChange={setMarital}
                    options={[
                      'Married',
                      'Single',
                      'Civil marriage',
                      'Separated',
                      'Widow',
                      'Unknown',
                    ]}
                    hint="UI reference only"
                  />
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="wizard-section">
                <div className="section-heading">
                  <span className="section-icon">
                    <Wallet size={19} />
                  </span>
                  <div>
                    <h3>Loan details</h3>
                    <p>
                      Provide the requested loan amount, income, repayment
                      terms, and loan characteristics.
                    </p>
                  </div>
                </div>
                <div className="fields-grid two">
                  <Num
                    label="Loan Amount Requested"
                    value={form.loan_amnt}
                    min={1}
                    onChange={(n) => updateNumber('loan_amnt', n)}
                    hint="Use the same currency units as the training dataset"
                  />
                  <Num
                    label="Monthly Payment (EMI)"
                    value={emi}
                    min={0}
                    onChange={setEmi}
                    hint="Used for the annual payment ratio only"
                  />
                  <Num
                    label="Annual Income"
                    value={form.annual_inc}
                    min={1}
                    onChange={(n) => updateNumber('annual_inc', n)}
                    hint="Annual gross income in dataset units"
                  />
                  <Num
                    label="Debt-to-Income Ratio (%)"
                    value={form.dti}
                    min={0}
                    max={99.99}
                    step={0.1}
                    onChange={(n) => updateNumber('dti', n)}
                    hint="Existing debt obligations relative to income"
                  />
                  <Sel
                    label="Loan Term"
                    value={form.term.trim()}
                    onChange={(s) =>
                      updateText('term', s === '36 months' ? ' 36 months' : ' 60 months')
                    }
                    options={['36 months', '60 months']}
                  />
                  <Sel
                    label="Loan Purpose"
                    value={form.purpose}
                    onChange={(s) => updateText('purpose', s)}
                    options={[
                      'debt_consolidation',
                      'credit_card',
                      'home_improvement',
                      'major_purchase',
                      'medical',
                      'small_business',
                      'car',
                      'other',
                    ]}
                  />
                  <Sel
                    label="Home Ownership"
                    value={form.home_ownership}
                    onChange={(s) => updateText('home_ownership', s)}
                    options={['RENT', 'MORTGAGE', 'OWN', 'OTHER', 'ANY']}
                  />
                  <Sel
                    label="Income Verification"
                    value={form.verification_status}
                    onChange={(s) => updateText('verification_status', s)}
                    options={['Verified', 'Source Verified', 'Not Verified']}
                  />
                </div>
              </div>
            )}

            {step === 3 && (
              <div className="wizard-section">
                <div className="section-heading">
                  <span className="section-icon">
                    <ShieldCheck size={19} />
                  </span>
                  <div>
                    <h3>Stability indicators</h3>
                    <p>Enter credit-history information used by your trained model.</p>
                  </div>
                </div>

                <div className="fields-grid two">
                  <Num
                    label="Delinquencies (Last 2 Years)"
                    value={form.delinq_2yrs}
                    onChange={(n) => updateNumber('delinq_2yrs', n)}
                  />
                  <Num
                    label="Open Credit Accounts"
                    value={form.open_acc}
                    onChange={(n) => updateNumber('open_acc', n)}
                  />
                  <Num
                    label="Public Records"
                    value={form.pub_rec}
                    onChange={(n) => updateNumber('pub_rec', n)}
                  />
                  <Num
                    label="Revolving Balance"
                    value={form.revol_bal}
                    onChange={(n) => updateNumber('revol_bal', n)}
                  />
                  <Num
                    label="Revolving Utilization (%)"
                    value={form.revol_util}
                    min={0}
                    max={100}
                    step={0.1}
                    onChange={(n) => updateNumber('revol_util', n)}
                  />
                  <Num
                    label="Total Credit Accounts"
                    value={form.total_acc}
                    onChange={(n) => updateNumber('total_acc', n)}
                  />
                  <Num
                    label="Days Since Phone Change"
                    value={stability[0]}
                    min={-10000}
                    max={0}
                    onChange={(n) =>
                      setStability((a) => a.map((v, i) => (i === 0 ? n : v)))
                    }
                    hint="Negative days indicate an event in the past"
                  />
                  <Num
                    label="Days Since ID Issued"
                    value={stability[1]}
                    min={-10000}
                    max={0}
                    onChange={(n) =>
                      setStability((a) => a.map((v, i) => (i === 1 ? n : v)))
                    }
                    hint="Negative days indicate an event in the past"
                  />
                  <Num
                    label="Days Since Registration"
                    value={stability[2]}
                    min={-10000}
                    max={0}
                    onChange={(n) =>
                      setStability((a) => a.map((v, i) => (i === 2 ? n : v)))
                    }
                    hint="Negative days indicate an event in the past"
                  />
                  <Num
                    label="Region Population (relative)"
                    value={stability[3]}
                    min={0}
                    max={0.1}
                    step={0.001}
                    onChange={(n) =>
                      setStability((a) => a.map((v, i) => (i === 3 ? n : v)))
                    }
                    hint="UI reference only"
                  />
                  <Sel
                    label="Region Credit Rating"
                    value={region}
                    onChange={setRegion}
                    options={['1', '2', '3']}
                    hint="UI reference only"
                  />
                  <Sel
                    label="Display Currency"
                    value={currency}
                    onChange={setCurrency}
                    options={['USD', 'INR']}
                    hint="Display only; does not convert model inputs"
                  />
                </div>

                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={includeExplanation}
                    onChange={(e) => setIncludeExplanation(e.target.checked)}
                  />
                  <span>Show plain-English explanation</span>
                </label>
                <p className="checkbox-note">
                  The explanation is a model summary, not financial advice or an
                  official lending decision.
                </p>
              </div>
            )}

            {error && (
              <div className="error-message">
                <AlertCircle size={17} /> {error}
              </div>
            )}

            <div className="wizard-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={() => setStep((current) => Math.max(0, current - 1))}
                disabled={step === 0 || loading}
              >
                <ArrowLeft size={17} /> Back
              </button>
              <span className="step-caption">Step {step + 1} of 4</span>
              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <LoaderCircle size={17} className="spin" /> Scoring…
                  </>
                ) : step === 3 ? (
                  <>
                    <ShieldCheck size={17} /> Score This Applicant
                  </>
                ) : (
                  <>
                    Next: {steps[step + 1].short} <ArrowRight size={17} />
                  </>
                )}
              </button>
            </div>
          </form>
        </section>
      )}

      {step === 4 && result && (
        <section className="results-section" id="results">
          <div className="results-heading">
            <div>
              <span className="overline">PREDICTION REPORT</span>
              <h2>Credit Risk Analysis</h2>
              <p>Results returned by your connected loan prediction API.</p>
            </div>
            <button className="secondary-button" onClick={startNewApplication}>
              <RotateCcw size={16} /> New Application
            </button>
          </div>

          <div className="score-range-card">
            <div className="score-range-heading">CREDIT SCORE</div>

           <div className={`score-range-value ${riskClass}`}>{score}</div>

            <div className="score-range-track">
              <div className="score-range-gradient" />
              <div
                className="score-range-marker"
                style={{ left: `${scorePct}%` }}
              />
            </div>

            <div className="score-range-labels">
              <span>{scoreMin} HIGH RISK</span>
              <span>{Math.round((scoreMin + scoreMax) / 2)}</span>
              <span>{scoreMax} LOW RISK</span>
            </div>

            <div className={`score-range-badge ${riskClass}`}>
              {riskBand.toUpperCase()}
            </div>

            <p className="score-range-note">
              Custom illustrative score — not an official FICO score.
            </p>
          </div>

          <div className="result-metrics">
            <article className="metric-card">
              <span className="metric-icon blue">
                <ChartNoAxesCombined size={20} />
              </span>
              <span className="metric-label">DEFAULT PROBABILITY</span>
              <strong>{(probability * 100).toFixed(2)}%</strong>
              <small>Model-estimated probability</small>
            </article>
            <article className="metric-card">
              <span className="metric-icon teal">
                <TrendingUp size={20} />
              </span>
              <span className="metric-label">ANNUAL PAYMENT RATIO</span>
              <strong>{annualPaymentRatio.toFixed(2)}%</strong>
              <small>Annualized EMI / income</small>
            </article>
            <article className="metric-card">
              <span className="metric-icon amber">
                <Wallet size={20} />
              </span>
              <span className="metric-label">EXPECTED LOSS</span>
              <strong>{money(expectedLoss)}</strong>
              <small>PD × exposure × assumed 45% LGD</small>
            </article>
          </div>

          <div className="results-grid">
            <section className="panel">
              <div className="panel-heading">
                <span className="section-icon">
                  <ChartNoAxesCombined size={18} />
                </span>
                <div>
                  <h3>Application summary</h3>
                  <p>Key values submitted for scoring</p>
                </div>
              </div>
              <div className="summary-list">
                {[
                  ['Loan amount', money(form.loan_amnt)],
                  ['Annual income', money(form.annual_inc)],
                  ['Monthly payment', money(emi)],
                  ['Annual payment ratio', `${annualPaymentRatio.toFixed(2)}%`],
                  ['Debt-to-income ratio', `${form.dti}%`],
                  ['Loan term', form.term.trim()],
                  ['Loan purpose', form.purpose.replace(/_/g, ' ')],
                  ['Home ownership', form.home_ownership],
                  ['Income verification', form.verification_status],
                ].map(([label, value]) => (
                  <div className="summary-row" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </section>

            <section className="panel">
              <div className="panel-heading">
                <span className="section-icon">
                  <ShieldCheck size={18} />
                </span>
                <div>
                  <h3>Risk assessment</h3>
                  <p>Interpretation of the model output</p>
                </div>
              </div>
              <div className={`risk-callout ${riskClass}`}>
                <span className="risk-callout-dot" />
                <div>
                  <strong>{riskBand}</strong>
                  <p>
                    Estimated default probability: {(probability * 100).toFixed(2)}%
                  </p>
                </div>
              </div>
              <p className="assessment-text">
                The model estimates the probability of the defined loan-default
                outcome from the supplied inputs. This estimate is not a
                guarantee of actual repayment behavior.
              </p>
              <div className="disclaimer-box">
                <AlertCircle size={17} />
                <p>
                  Research use only. Do not use this score alone to approve or
                  deny a loan.
                </p>
              </div>
            </section>
          </div>

          <section className="panel explanation-panel">
            <div className="panel-heading">
              <span className="section-icon">
                <TrendingUp size={18} />
              </span>
              <h3>Decision Factors (SHAP)</h3>
              <p>
                The features that most influenced this model's default-risk
                prediction. Positive SHAP values increase the model output;
                negative values decrease it.
              </p>
            </div>
            {topFactors.length ? (
              <div className="factor-list">
                {topFactors.map((factor, index) => {
                  const positive = factor.impact > 0;
                  const width = Math.max(
                    8,
                    (Math.abs(factor.impact) / maxImpact) * 100,
                  );
                  return (
                    <div className="factor-row" key={`${factor.feature}-${index}`}>
                      <div className="factor-name">
                        {names[factor.feature] ?? factor.feature.replace(/_/g, ' ')}
                      </div>
                      <div className="factor-track">
                        <div
                          className={`factor-fill ${positive ? 'positive' : 'negative'}`}
                          style={{ width: `${width}%` }}
                        />
                      </div>
                      <span
                        className={`factor-impact ${positive ? 'positive-text' : 'negative-text'}`}
                      >
                        {positive ? 'Increases' : 'Decreases'} model output
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="empty-state">
                The API did not return decision-factor data for this prediction.
              </p>
            )}
            <p className="legal-note">
              These research factors are not legally reviewed adverse-action
              reasons. Regulatory notices require separate validation.
            </p>
          </section>

          {includeExplanation && (
            <section className="panel explanation-panel">
              <div className="panel-heading">
                <span className="section-icon">
                  <Sparkles size={18} />
                </span>
                <div>
                  <h3>Plain-English explanation</h3>
                  <p>Summary of the model result</p>
                </div>
              </div>
              <div className="plain-english">
                {result.explanation ??
                  `The model estimated a ${(probability * 100).toFixed(2)}% probability of the defined loan-default outcome. The custom score is ${score} on a ${scoreMin}–${scoreMax} scale (${riskBand.toLowerCase()}). The estimate reflects patterns learned from historical LendingClub data and should not be interpreted as a guarantee or an official credit score.`}
              </div>
            </section>
          )}

          <button
            className="primary-button new-application-button"
            onClick={startNewApplication}
          >
            <RotateCcw size={17} /> Start New Application
          </button>
        </section>
      )}

      <footer className="footer">
        <span className="footer-brand">
          <ChartNoAxesCombined size={17} /> Loan Default Intelligence
        </span>
        <span>React + TypeScript · FastAPI · LendingClub historical loan data</span>
        <span>Research demonstration · Not an official credit score</span>
      </footer>
    </main>
  );
}