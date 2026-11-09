"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import bsa_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)", separators=",.")
    return lock_axes(fig)


def build_allocation(a):
    """Ersparnis je Mitglied (Prozent der Einzelkosten) je Aufteilungsregel; die Querstriche markieren den Kern-Spielraum (kleinste und größte Ersparnis im Kern)."""
    labels = [f"Lager {i + 1}" for i in range(a.game.n)]
    fig = go.Figure()
    for rule in C.ALLOCATION_RULES:
        fig.add_trace(go.Bar(x=labels, y=[100 * s for s in a.savings[rule]], name=rule, marker_color=C.COLOR_RULES[rule],
                             hovertemplate="%{x}, " + rule + ": %{y:.1f} %<extra></extra>"))
    if a.core_lo is not None:
        single = a.game.single
        best = [100 * (single[i] - a.core_lo[i]) / single[i] for i in range(a.game.n)]
        worst = [100 * (single[i] - a.core_hi[i]) / single[i] for i in range(a.game.n)]
        for name, ys in (("Kern: größte mögliche Ersparnis", best), ("Kern: kleinste mögliche Ersparnis", worst)):
            fig.add_trace(go.Scatter(x=labels, y=ys, mode="markers", name=name, marker=dict(symbol="line-ew", size=34, line=dict(width=3, color=C.COLOR_CORE)),
                                     hovertemplate="%{x}, " + name + ": %{y:.1f} %<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Ersparnis gegenüber allein (% der Einzelkosten)")
    return _base(fig)


def build_gain_heatmap(rows):
    """Mittlerer Gewinn des Bündelns (Prozent der Einzelkosten) je Lagerzahl und Korrelation."""
    ns = sorted({r["n"] for r in rows})
    rhos = sorted({r["rho"] for r in rows})
    cell = {(r["n"], r["rho"]): r["mean"] for r in rows}
    z = [[100 * cell[(n, rho)] for rho in rhos] for n in ns]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"ρ = {rho:.2f}".replace(".", ",") for rho in rhos], y=[f"n = {n}" for n in ns], colorscale="Blues", zmin=0,
        texttemplate="%{z:.1f} %", hovertemplate="%{y}, %{x}: %{z:.1f} %<extra></extra>", colorbar=dict(title=dict(text="Gewinn (%)", side="top"), nticks=4),
    ))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category")
    return _base(fig, 360)


def build_rule_counts(counts, instances):
    """Anteil der Zufallsinstanzen, in denen die Aufteilung im Kern liegt, je Regel."""
    rules = list(C.ALLOCATION_RULES)
    fig = go.Figure(go.Bar(
        x=rules, y=[100 * counts[r] / instances if instances else 0 for r in rules], marker_color=[C.COLOR_RULES[r] for r in rules],
        text=[f"{counts[r]} von {instances}" for r in rules], textposition="outside", hovertemplate="%{x}: %{y:.1f} %<extra></extra>",
    ))
    fig.update_yaxes(title_text="Instanzen mit Aufteilung im Kern (%)", range=[0, 118])
    return _base(fig, 340)


def build_exit(summary):
    """Anteil der Instanzen mit Einzelaustritt je Korrelationsklasse."""
    labels = [f"ρ {b['lo']:.1f} bis {b['hi']:.1f}".replace(".", ",") for b in summary["by_rho"]]
    share = [100 * b["exits"] / b["instances"] if b["instances"] else 0 for b in summary["by_rho"]]
    fig = go.Figure(go.Bar(
        x=labels, y=share, marker_color=C.COLOR_LINE, text=[f"{b['exits']} von {b['instances']}" for b in summary["by_rho"]], textposition="outside",
        hovertemplate="%{x}: %{y:.1f} %<extra></extra>",
    ))
    fig.update_yaxes(title_text="Instanzen mit Einzelaustritt (%)", range=[0, 118])
    return _base(fig, 320)
