

import os
import textwrap


def _build_prompt(summary: dict, traffic_df, distance_corr: float, combo_df, insights: list) -> str:
    """
    Assemble a compact, structured prompt out of the analysis results.
    Keeping this as its own function makes it easy to tweak the wording
    later without touching the API-calling logic.
    """
    traffic_lines = "\n".join(
        f"  - {row.Road_traffic_density}: avg {row.avg_delivery_time} min "
        f"(median {row.median_delivery_time} min, {row.order_count} orders)"
        for row in traffic_df.itertuples()
    )

    combo_lines = "\n".join(
        f"  - {row.Weather_conditions} + {row.Road_traffic_density}: "
        f"avg {row.avg_delivery_time} min ({row.order_count} orders)"
        for row in combo_df.head(5).itertuples()
    )

    prompt = textwrap.dedent(f"""
        You are a data analyst writing a short summary for a restaurant
        operations manager who is not technical. Explain the findings
        below in plain, friendly English. Keep it to 3-4 short
        paragraphs, avoid jargon, and focus on what the manager should
        actually do differently based on this data. Do not invent any
        numbers beyond what's given here.

        Dataset overview:
          - Total orders analyzed: {summary['total_orders']}
          - Average delivery time: {summary['avg_delivery_time_min']} minutes
          - Average delivery distance: {summary['avg_distance_km']} km

        Delivery time by traffic level:
        {traffic_lines}

        Correlation between distance and delivery time: {distance_corr}

        Five slowest weather + traffic combinations:
        {combo_lines}

        Pre-computed insights to weave in naturally:
        {chr(10).join('- ' + i for i in insights)}
    """).strip()

    return prompt


def _fallback_summary(insights: list) -> str:
    """
    A dependency-free explanation used when the Gemini API isn't
    reachable (missing key, network issue, quota, etc). It's less
    polished than the AI-generated version, but it means the pipeline
    never hard-fails just because of an external service.
    """
    intro = (
        "Here's a plain-language summary of what the delivery data shows "
        "(generated locally - the AI explanation service wasn't available):\n"
    )
    body = "\n".join(f"- {point}" for point in insights)
    return intro + body


def explain_findings(summary: dict, traffic_df, distance_corr: float, combo_df, insights: list) -> str:
    """
    Main entry point: takes the analysis outputs and returns a
    plain-English explanation string, using Gemini when available.

    We deliberately don't raise on failure - a broken API key shouldn't
    stop someone from getting their charts and CSVs.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _fallback_summary(insights)


    candidate_models = [
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-3.7-flash",
        "gemini-2.5-flash",
    ]

    try:
        from google import genai
    except ImportError as error:
        print(f"[ai_explanation] google-genai package not installed ({error}); using fallback summary.")
        return _fallback_summary(insights)

    prompt = _build_prompt(summary, traffic_df, distance_corr, combo_df, insights)
    client = genai.Client(api_key=api_key)

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            text = getattr(response, "text", None)
            if text:
                return text.strip()
        except Exception as error:
            print(f"[ai_explanation] Gemini model '{model_name}' failed ({error}); trying next option.")
            continue

    print("[ai_explanation] All Gemini model attempts failed; using fallback summary.")
    return _fallback_summary(insights)