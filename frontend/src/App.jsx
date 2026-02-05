import { useMemo, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MAX_MISSING_FIELDS = 3;

const fieldGroups = [
  {
    title: "Patient Basics",
    description: "Demographics and residence information.",
    fields: [
      {
        name: "gender",
        label: "Gender",
        type: "select",
        options: ["Female", "Male", "Other"],
        required: true
      },
      {
        name: "age",
        label: "Age",
        type: "number",
        min: 0,
        max: 120,
        step: "1",
        placeholder: "e.g. 67",
        required: true
      },
      {
        name: "Residence_type",
        label: "Residence type",
        type: "select",
        options: ["Urban", "Rural"],
        required: true
      }
    ]
  },
  {
    title: "Clinical Indicators",
    description: "Known clinical risk factors.",
    fields: [
      {
        name: "hypertension",
        label: "Hypertension",
        type: "select",
        options: ["0", "1"],
        helper: "0 = No, 1 = Yes",
        required: true
      },
      {
        name: "heart_disease",
        label: "Heart disease",
        type: "select",
        options: ["0", "1"],
        helper: "0 = No, 1 = Yes",
        required: true
      },
      {
        name: "avg_glucose_level",
        label: "Average glucose level",
        type: "number",
        min: 30,
        max: 300,
        step: "0.1",
        placeholder: "e.g. 110.5",
        required: true
      },
      {
        name: "bmi",
        label: "BMI",
        type: "number",
        min: 10,
        max: 60,
        step: "0.1",
        placeholder: "e.g. 28.4",
        required: false
      }
    ]
  },
  {
    title: "Lifestyle",
    description: "Work and lifestyle history.",
    fields: [
      {
        name: "ever_married",
        label: "Ever married",
        type: "select",
        options: ["Yes", "No"],
        required: true
      },
      {
        name: "work_type",
        label: "Work type",
        type: "select",
        options: ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        required: true
      },
      {
        name: "smoking_status",
        label: "Smoking status",
        type: "select",
        options: ["never smoked", "formerly smoked", "smokes", "Unknown"],
        required: false
      }
    ]
  }
];

const initialState = fieldGroups
  .flatMap((group) => group.fields)
  .reduce((acc, field) => {
    acc[field.name] = "";
    return acc;
  }, {});

function classNames(...values) {
  return values.filter(Boolean).join(" ");
}

function calculateRiskScore(probability) {
  // Convert probability to risk score (0-100%)
  // Apply non-linear scaling to spread values more intuitively
  const scaled = Math.pow(probability, 0.7) * 100;
  return Math.min(100, scaled).toFixed(1);
}

export default function App() {
  const [formState, setFormState] = useState(initialState);
  const [errors, setErrors] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);

  const missingFields = useMemo(() => {
    return Object.entries(formState)
      .filter(([, value]) => value === "")
      .map(([key]) => key);
  }, [formState]);

  const handleChange = (fieldName, value) => {
    setFormState((prev) => ({ ...prev, [fieldName]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setErrors("");
    setResponse(null);

    if (missingFields.length > MAX_MISSING_FIELDS) {
      setErrors(
        `Please fill more fields. At most ${MAX_MISSING_FIELDS} fields can be left blank.`
      );
      return;
    }

    const payload = Object.entries(formState).reduce((acc, [key, value]) => {
      acc[key] = value === "" ? null : value;
      return acc;
    }, {});

    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        setErrors(data.error || "Prediction failed.");
      } else {
        setResponse(data);
      }
    } catch (error) {
      setErrors("Unable to reach the prediction service.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-5 py-10 text-ink-900">
      <div className="mx-auto max-w-6xl">
        <header className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-ink-500">
              Hospital Intake
            </p>
            <h1 className="font-display text-4xl text-ink-900 md:text-5xl">
              Stroke Risk Assessment
            </h1>
            <p className="mt-3 max-w-2xl text-base text-ink-600">
              Enter patient information from the clinical dataset. All categories
              are recommended for the most reliable prediction. Up to {MAX_MISSING_FIELDS}
              fields may be left blank and will be imputed with dataset medians or
              modes.
            </p>
          </div>
          <div className="rounded-2xl border border-white/70 bg-white/80 px-4 py-3 shadow-sm backdrop-blur">
            <p className="text-xs uppercase tracking-[0.2em] text-ink-400">
              Model Suite
            </p>
            <p className="text-sm font-semibold text-ink-700">
              Logistic Regression, Random Forest, Neural Network
            </p>
          </div>
        </header>

        <main className="mt-10 grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
          <section className="rounded-3xl border border-white/70 bg-white/80 p-6 shadow-lg backdrop-blur card-glow">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-2xl text-ink-900">Patient Intake</h2>
              <span className="rounded-full bg-ember-100 px-3 py-1 text-xs font-semibold text-ember-700">
                Missing {missingFields.length}/{MAX_MISSING_FIELDS}
              </span>
            </div>
            <form className="mt-6 space-y-8" onSubmit={handleSubmit}>
              {fieldGroups.map((group) => (
                <div key={group.title} className="space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold text-ink-800">
                      {group.title}
                    </h3>
                    <p className="text-sm text-ink-500">{group.description}</p>
                  </div>
                  <div className="grid gap-4 md:grid-cols-2">
                    {group.fields.map((field) => (
                      <label
                        key={field.name}
                        className="flex flex-col gap-2 rounded-2xl border border-ink-100 bg-ink-50/40 px-4 py-3"
                      >
                        <span className="text-sm font-semibold text-ink-700">
                          {field.label}
                          {field.required && (
                            <span className="ml-2 text-xs font-semibold uppercase text-tide-600">
                              Recommended
                            </span>
                          )}
                        </span>
                        {field.type === "select" ? (
                          <select
                            value={formState[field.name]}
                            onChange={(event) =>
                              handleChange(field.name, event.target.value)
                            }
                            className="rounded-xl border border-ink-200 bg-white px-3 py-2 text-sm text-ink-700 focus:border-tide-400 focus:outline-none"
                          >
                            <option value="">Select</option>
                            {field.options.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type={field.type}
                            min={field.min}
                            max={field.max}
                            step={field.step}
                            placeholder={field.placeholder}
                            value={formState[field.name]}
                            onChange={(event) =>
                              handleChange(field.name, event.target.value)
                            }
                            className="rounded-xl border border-ink-200 bg-white px-3 py-2 text-sm text-ink-700 focus:border-tide-400 focus:outline-none"
                          />
                        )}
                        {field.helper && (
                          <span className="text-xs text-ink-400">{field.helper}</span>
                        )}
                      </label>
                    ))}
                  </div>
                </div>
              ))}

              {errors && (
                <div className="rounded-2xl border border-ember-200 bg-ember-50 px-4 py-3 text-sm text-ember-700">
                  {errors}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className={classNames(
                  "w-full rounded-2xl px-6 py-3 text-sm font-semibold uppercase tracking-[0.25em]",
                  loading
                    ? "cursor-not-allowed bg-ink-300 text-white"
                    : "bg-ink-900 text-white hover:bg-ink-800"
                )}
              >
                {loading ? "Running models..." : "Generate Risk Report"}
              </button>
            </form>
          </section>

          <section className="space-y-6">
            <div className="rounded-3xl border border-white/70 bg-white/90 p-6 shadow-lg backdrop-blur card-glow">
              <h2 className="font-display text-2xl text-ink-900">Model Output</h2>
              <p className="mt-2 text-sm text-ink-500">
                Predictions and most influential feature from each model.
              </p>

              {response ? (
                <div className="mt-6 space-y-4">
                  {Object.entries(response.predictions).map(([modelKey, value]) => (
                    <div
                      key={modelKey}
                      className="rounded-2xl border border-ink-100 bg-ink-50/40 px-4 py-3"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-xs uppercase tracking-[0.2em] text-ink-400">
                            {modelKey.replace(/_/g, " ")}
                          </p>
                          <p className="text-lg font-semibold text-ink-800">
                            {value.risk} risk
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-xs uppercase tracking-[0.2em] text-ink-400">
                            Risk Score
                          </p>
                          <p className="text-lg font-semibold text-tide-700">
                            {calculateRiskScore(value.probability)}%
                          </p>
                        </div>
                      </div>
                      <p className="mt-2 text-sm text-ink-600">
                        Primary driver: {response.top_features[modelKey]}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-6 rounded-2xl border border-dashed border-ink-200 bg-white px-4 py-6 text-sm text-ink-500">
                  Submit patient data to view model output.
                </div>
              )}
            </div>

            <div className="rounded-3xl border border-white/70 bg-white/90 p-6 shadow-lg backdrop-blur card-glow">
              <h2 className="font-display text-2xl text-ink-900">Clinical Explanation</h2>
              {response ? (
                <div className="mt-4 space-y-3">
                  <div className="rounded-2xl border border-tide-200 bg-tide-50 px-4 py-3">
                    <p className="text-sm font-semibold text-tide-700">
                      Most influential feature
                    </p>
                    <p className="text-lg font-semibold text-ink-800">
                      {response.most_influential_feature.feature}
                    </p>
                    <p className="text-sm text-ink-600">
                      {response.most_influential_feature.note}
                    </p>
                  </div>
                  {response.missing_fields.length > 0 && (
                    <div className="rounded-2xl border border-ember-200 bg-ember-50 px-4 py-3 text-sm text-ember-700">
                      Missing fields were imputed: {response.missing_fields.join(", ")}
                    </div>
                  )}
                </div>
              ) : (
                <p className="mt-4 text-sm text-ink-500">
                  Explanation appears after model evaluation.
                </p>
              )}
            </div>
          </section>
        </main>

        <footer className="mt-10 text-xs text-ink-500">
          This tool supports clinical decision-making and must be validated by a
          qualified professional before use in care pathways.
        </footer>
      </div>
    </div>
  );
}
