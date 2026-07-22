import "./App.css";
import { useState, useEffect, useRef } from "react";
import PlayerForm from "./PlayerForm";

function inchesToFeetInches(inches) {
  const feet = Math.floor(inches / 12);
  const newInches = inches - feet * 12;
  return `${feet}'${newInches}"`;
}





function SkillInput({ label, value, onChange }) {
  console.log("SkillInput value:", value, typeof value);

  return (
    
    <div className="skill-input">
      <label>{label}</label>
      <img
        className="skill-image"
        src={`/SkillLevelGif/skill_level_${value}.gif`}
        style={{
          width: "123px",
          height: "9px",
          backgroundColor: "white", // behind the image
          textAlign: "center",
        }}
        alt={`Skill level ${value}`}
        
      />
      <input
        type="number"
        min="0"
        max="20"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      
      
    </div>
  );
  
}



function App() {
  const [playerUrl, setPlayerUrl] = useState("");
  const [playerName, setPlayerName] = useState("");

  // Fetches Player Url
  const handleKeyDown = (e) => {
    console.log("Submitted URL:", playerUrl, playerYear);
    fetchPlayerUrl(playerUrl, playerYear);
  };

  const fetchPlayerUrl = async (player_link, playerYear) => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:2000/player-url-submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: player_link, player_year: playerYear }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.json();
      console.log("Flask response:", result);

      const player_url_skills = result.player_skills;

      setPlayerName(player_url_skills.Name);
      // Set Player measurables
      setMeasurables({
        height: player_url_skills.height,
        wingspan: player_url_skills.wingspan,
        weight: player_url_skills.weight,
        vertical: player_url_skills.vertical,
      });

      //Set Player skills
      setSkills({
        insideShot: player_url_skills.IS,
        basketballIQ: player_url_skills.IQ,
        outsideShot: player_url_skills.OS,
        passing: player_url_skills.Pass,
        shootingRange: player_url_skills.Rng,
        ballHandling: player_url_skills.Hnd,
        finishing: player_url_skills.Fin,
        driving: player_url_skills.Drv,
        rebounding: player_url_skills.Reb,
        strength: player_url_skills.Str,
        interiorDefense: player_url_skills.IDef,
        speed: player_url_skills.Spd,
        perimeterDefense: player_url_skills.PDef,
        stamina: player_url_skills.Sta,
      });
    } catch (error) {
      console.error("Error sending to Flask:", error);
    } finally {
      if (isMounted.current) setIsLoading(false);
    }
  };

  const [skills, setSkills] = useState({
    insideShot: 10,
    basketballIQ: 10,
    outsideShot: 10,
    passing: 10,
    shootingRange: 10,
    ballHandling: 10,
    finishing: 10,
    driving: 10,
    rebounding: 10,
    strength: 10,
    interiorDefense: 10,
    speed: 10,
    perimeterDefense: 10,
    stamina: 10,
  });

  const skill_index = Object.values(skills).reduce((sum, val) => sum + val, 0);

  const [measurables, setMeasurables] = useState({
    height: 75,
    weight: 200,
    wingspan: 70,
    vertical: 30,
  });

  const [position, setPosition] = useState("PG");
  const [flaskData, setFlaskData] = useState(null);
  const [previousFlaskData, setPreviousFlaskData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const [playerYear, setPlayerYear] = useState("Current");

  const isMounted = useRef(false);
  const debounceTimer = useRef(null);

  const setMeasurable = (key, value) => {
    setMeasurables((prev) => ({ ...prev, [key]: value }));
  };

  const handleSkillChange = (key, value) => {
    setSkills((prev) => ({ ...prev, [key]: value }));
  };

  const cap20 = v => Math.min(v, 20);

  const exportToJSON = () => {
    return {
      Position: position,
      IS: skills.insideShot,
      IQ: skills.basketballIQ,
      OS: skills.outsideShot,
      Pass: skills.passing,
      Rng: skills.shootingRange,
      Hnd: skills.ballHandling,
      Fin: skills.finishing,
      Drv: skills.driving,
      Reb: skills.rebounding,
      Str: skills.strength,
      IDef: skills.interiorDefense,
      Spd: skills.speed,
      PDef: skills.perimeterDefense,
      Sta: skills.stamina,
      height: measurables.height,
      wingspan: measurables.wingspan,
      weight: measurables.weight,
      vertical: measurables.vertical,
    };
  };

  
  const fetchDataFromFlask = async (dataToSend) => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:2000/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(dataToSend),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.json();
      console.log("Flask response:", result);

      setPreviousFlaskData(flaskData); // Save current as previous
      setFlaskData(result.received); // Update with new data
    } catch (error) {
      console.error("Error sending to Flask:", error);
    } finally {
      if (isMounted.current) setIsLoading(false);
    }
  };

  useEffect(() => {
  if (playerName && position) {
    document.title = `${playerName} - ${position}`;
  } else {
    document.title = "Player Skill Evaluator"; // fallback title
  }
  }, [playerName, position]);

  // Initial fetch on mount (optional, can keep or remove)
  useEffect(() => {
    isMounted.current = true;

    // You can fetch initial data here if you want
    // Or just rely on auto-fetch on input change

    // Cleanup on unmount
    return () => {
      isMounted.current = false;
    };
  }, []);

  // Auto fetch on skills, measurables, or position change with debounce
  useEffect(() => {
    if (!isMounted.current) return;

    if (debounceTimer.current) clearTimeout(debounceTimer.current);

    debounceTimer.current = setTimeout(() => {
      const data = exportToJSON();
      fetchDataFromFlask(data);
    }, 300); // 500ms debounce delay

    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
    };
  }, [skills, measurables, position]);

  // Auto-fetch when playerYear changes AND a playerUrl exists
  useEffect(() => {
    if (playerUrl.trim() === "") return;

    const timer = setTimeout(() => {
      fetchPlayerUrl(playerUrl, playerYear);
    }, 300); // 500ms delay

    return () => clearTimeout(timer); // Clear timeout on cleanup
  }, [playerYear]);

  return (
    <>
      <div className="app-container">
        <h1 className="page-title">Player Skill Evaluator</h1>
        <div className="stat-table">
          <table>
            <thead>
              <tr></tr>
            </thead>
            <tbody>
              <tr>
                <td>Age: N/A</td>
                <td>
                  <SkillInput
                    label="Inside Shot:"
                    value={cap20(skills.insideShot)}
                    onChange={(val) => handleSkillChange("insideShot", val)}
                  />
                </td>
                <td>
                  <SkillInput
                    label="Basketball IQ:"
                    value={cap20(skills.basketballIQ)}
                    onChange={(val) => handleSkillChange("basketballIQ", val)}
                  />
                </td>
              </tr>
              <tr>
                <td>Shoots: N/A</td>
                <td>
                  <SkillInput
                    label="Outside Shot:"
                    value={cap20(skills.outsideShot)}
                    onChange={(val) => handleSkillChange("outsideShot", val)}
                  />
                </td>
                <td>
                  <SkillInput
                    label="Passing:"
                    value={cap20(skills.passing)}
                    onChange={(val) => handleSkillChange("passing", val)}
                  />
                </td>
              </tr>
              <tr>
                <td>
                  <label htmlFor="height">Height:</label>
                  <input
                    type="number"
                    id="height"
                    min="65"
                    max="90"
                    step="0.5"
                    value={measurables.height}
                    onChange={(e) =>
                      setMeasurable("height", Number(e.target.value))
                    }
                  />
                  in <br />
                  <span>{inchesToFeetInches(measurables.height)}</span>
                </td>
                <td>
                  <SkillInput
                    label="Shooting Range:"
                    value={cap20(skills.shootingRange)}
                    onChange={(val) => handleSkillChange("shootingRange", val)}
                  />
                </td>
                <td>
                  <SkillInput
                    label="Ball Handling:"
                    value={cap20(skills.ballHandling)}
                    onChange={(val) => handleSkillChange("ballHandling", val)}
                  />
                </td>
              </tr>
              <tr>
                <td>
                  <label htmlFor="weight">Weight:</label>
                  <input
                    type="number"
                    id="weight"
                    min="100"
                    max="350"
                    step="5"
                    value={measurables.weight}
                    onChange={(e) =>
                      setMeasurable("weight", Number(e.target.value))
                    }
                  />{" "}
                  lbs
                </td>
                <td>
                  <SkillInput
                    label="Finishing:"
                    value={cap20(skills.finishing)}
                    onChange={(val) => handleSkillChange("finishing", val)}
                  />
                </td>
                <td>
                  <SkillInput
                    label="Driving:"
                    value={cap20(skills.driving)}
                    onChange={(val) => handleSkillChange("driving", val)}
                  />
                </td>
              </tr>
              <tr>
                <td>
                  <label htmlFor="wingspan">Wingspan:</label>
                  <input
                    type="number"
                    id="wingspan"
                    min="65"
                    max="96"
                    step="0.25"
                    value={measurables.wingspan}
                    onChange={(e) =>
                      setMeasurable("wingspan", Number(e.target.value))
                    }
                  />
                  in <br />
                  <span>{inchesToFeetInches(measurables.wingspan)}</span>
                </td>
                <td>
                  <SkillInput
                    label="Rebounding:"
                    value={skills.rebounding}
                    onChange={(val) => handleSkillChange("rebounding", val)}
                  />
                </td>
                <td>
                  <SkillInput
                    label="Strength:"
                    value={cap20(skills.strength)}
                    onChange={(val) => handleSkillChange("strength", val)}
                  />
                </td>
              </tr>
              <tr>
                <td>
                  <label htmlFor="vertical">Vertical:</label>
                  <input
                    type="number"
                    id="vertical"
                    min="20"
                    max="50"
                    step="0.5"
                    value={measurables.vertical}
                    onChange={(e) =>
                      setMeasurable("vertical", Number(e.target.value))
                    }
                  />
                  in
                </td>
                <td>
                  <SkillInput
                    label="Interior Defense:"
                    value={cap20(skills.interiorDefense)}
                    onChange={(val) =>
                      handleSkillChange("interiorDefense", val)
                    }
                  />
                </td>
                <td>
                  <SkillInput
                    label="Speed:"
                    value={cap20(skills.speed)}
                    onChange={(val) => handleSkillChange("speed", val)}
                  />
                </td>
              </tr>
              <tr>
                <td className="highlight" />
                <td className="highlight">
                  <SkillInput
                    label="Perimeter Defense:"
                    value={cap20(skills.perimeterDefense)}
                    onChange={(val) =>
                      handleSkillChange("perimeterDefense", val)
                    }
                  />
                </td>
                <td className="highlight">
                  <SkillInput
                    label="Stamina:"
                    value={cap20(skills.stamina)}
                    onChange={(val) => handleSkillChange("stamina", val)}
                  />
                </td>
              </tr>
              <tr>
                <td className="highlight" />
                <td className="highlight" />
                <td className="highlight">
                  <div className="skill-input">
                    <label>Skill Index:</label>
                    <img
                      className="skill-image"
                      src={`/SkillLevelGif/skill_level_${Math.ceil(
                        skill_index / 15
                      )}.gif`}
                      style={{
                        width: "123px",
                        height: "9px",
                        backgroundColor: "white", // 👈 background behind the image,
                      }}
                      alt={`Skill index ${skill_index}`}
                    ></img>
                    <input
                      type="number"
                      value={skill_index}
                      style={{ width: "70px" }}
                      readOnly // or remove this if user should be able to manually change it
                    />
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
          <p className="position-label">Select Player Position</p>
          <div className="radio-group">
            {["PG", "SG", "SF", "PF", "C"].map((pos) => (
              <label key={pos}>
                <input
                  type="radio"
                  name="position"
                  value={pos}
                  checked={position === pos}
                  onChange={() => setPosition(pos)}
                />
                {pos}
              </label>
            ))}
          </div>
          <input
            type="url"
            value={playerUrl}
            onChange={(e) => setPlayerUrl(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter URL and press Enter"
            className="border p-2 rounded"
          />
          <div className="radio-group">
            {["Current", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"].map(
              (wanted_player_year) => (
                <label key={wanted_player_year}>
                  <input
                    type="radio"
                    name="playerYear"
                    value={wanted_player_year}
                    checked={playerYear === wanted_player_year}
                    onChange={() => setPlayerYear(wanted_player_year)}
                  />
                  {wanted_player_year}
                </label>
              )
            )}
          </div>
        </div>

        <div className="playerform-wrapper">
          {isLoading ? (
            <p>Loading...</p>
          ) : (
            <PlayerForm
              stats={flaskData}
              previousStats={previousFlaskData}
              playerName={playerName}
              playerUrl={playerUrl}
              playerYear={playerYear}
              position={position}
            />
          )}
        </div>
      </div>
    </>
  );
}

export default App;
