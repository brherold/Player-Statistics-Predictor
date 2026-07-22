import React from "react";
import "./PlayerForm.css";

function lerp(start, end, t) {
  return Math.round(start + (end - start) * t);
}

function getArrowColor(diff, isUp) {
  // Normalize diff magnitude, cap max at 50 for scaling
  const t = Math.min(Math.abs(diff) / 50, 1);

  // Gray base color
  const gray = { r: 70, g: 70, b: 75 };

  // Target bright colors
  const brightGreen = { r: 0, g: 255, b: 50 };
  const brightRed = { r: 255, g: 50, b: 50 };

  const target = isUp ? brightGreen : brightRed;

  // Interpolate each color channel based on t
  const r = lerp(gray.r, target.r, t);
  const g = lerp(gray.g, target.g, t);
  const b = lerp(gray.b, target.b, t);

  return `rgb(${r},${g},${b})`;
}

export default function PlayerForm({
  stats,
  previousStats,
  playerName,
  playerUrl,
  playerYear,
  position,
}) {
  const statKeys = [
    "Fin%",
    "IS%",
    "Mid%",
    "3PT%",
    "FT%",
    "Reb",
    "Ast",
    "Stl",
    "Blk",
    "FD",
    "Ast/TO",
    "O2%",
    "O3%",
    "ORAPM",
    "DRAPM",
    "RAPM",
    "",
    "OPM+",
    "DPM+",
    "EPM+"
  ];
  console.log("Current stats:", stats);
  console.log("Previous stats:", previousStats);

  return (
    <div id="main">
      <div id="title">
        <h1>Hardwood Player Stat Predictor</h1>
        <h1 className="player-name">
        <a
          href={
            playerUrl.includes("/")
              ? `https://onlinecollegebasketball.org/player/${playerUrl.split("/").pop()}`
              : `https://onlinecollegebasketball.org/player/${playerUrl}`
          }
          target="_blank"
          rel="noopener noreferrer"
        >
          {playerName}
        </a>
        </h1>
        <p>Position: {position}</p>
        <p>Predicted Year: {playerYear}</p>
      </div>
      <div className="table-wrapper">
        <table className="stats-table">
          <thead>
            <tr>
              {statKeys.map((stat) => {
                if (stat === "OPM+") {
                  return (
                    <th
                      key={stat}
                      title="Offensive Plus-Minus: Measures a player's offensive impact per 100 possessions above the average player."
                    >
                      {stat}
                      <span>ℹ️</span>
                    </th>
                  );
                } else if (stat === "DPM+") {
                  return (
                    <th
                      key={stat}
                      title="Defensive Plus-Minus: Measures a player's defensive impact per 100 possessions above the average player."
                    >
                      {stat}
                      <span>ℹ️</span>
                    </th>
                  );
                } else if (stat === "EPM+") {
                  return (
                    <th
                      key={stat}
                      title="Estimated Plus-Minus: Measures a player's total impact per 100 possessions above the average player."
                    >
                      {stat}
                      <span>ℹ️</span>
                    </th>
                  );
                } else {
                  return <th key={stat}>{stat}</th>;
                }
              })}
            </tr>
          </thead>
          <tbody>
            <tr>
              {statKeys.map((key) => {
                const stat = stats ? stats[key] : null;
                const prev = previousStats ? previousStats[key] : null;

                let displayValue = stat?.value ?? "N/A";
                const statValue = stat?.value;

                let arrow = null;
                const statPercentile = parseFloat(stat?.percentile);
                const prevPercentile = parseFloat(prev?.percentile);

                if (!isNaN(statPercentile) && !isNaN(prevPercentile)) {
                  const diff = statPercentile - prevPercentile;

                  if (diff !== 0) {
                    const color = getArrowColor(diff, diff > 0);

                    arrow = (
                      <span
                        style={{
                          color: color,
                          fontSize: "15px",
                          marginLeft: "4px",
                        }}
                        title={`Percentile change: ${
                          diff > 0 ? "+" : ""
                        }${diff.toFixed(1)}%`}
                      >
                        {diff > 0 ? "↑" : "↓"}
                      </span>
                    );
                  }
                }

                if (key.includes("PM") && statValue > 0) {
                  displayValue = `+${statValue}`;
                }

                return (
                  <td key={key} className="stat-cell">
                    <div className="stat-value">
                      {displayValue} {arrow}
                    </div>
                    <div
                      className="stat-percentile"
                      style={{
                        color: stat?.color ?? "black",
                        fontWeight: "bold",
                        fontSize: "11px",
                        marginRight: "10px",
                      }}
                    >
                      {stat?.percentile ?? "-"}
                    </div>
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      </div>
      <div className="per-minutes">
        <b>Per 30 Mins</b>
      </div>
      <div class="gradient-bar-container">
        <div class="gradient-labels">
          <span class="gradient-label-left">Bad</span>
          <span class="gradient-label-right">Good</span>
        </div>
        <div class="gradient-bar"></div>
      </div>
    </div>
  );
}
