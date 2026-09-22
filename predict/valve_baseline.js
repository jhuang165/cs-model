"use strict";
// Walk-forward evaluation of Valve's ranking model (model/ranking.js), driven by valve_baseline.py.
//
//   node predict/valve_baseline.js <matchdata.json> <evalFrom> <evalTo> <stepSeconds>
//
// For every step start T it rebuilds Valve's standings with data up to T (generateRanking's own
// six-month window and seeding), then prices every match that starts in [T, T+step) from the
// rank values of the two rosters, exactly as model/fit.js does (win/loss delta ratio at fixed
// RD 75). Prints one line per match: matchStartTime team1Id team2Id p   (p = -1 when a roster
// is not in the standings).

const path = require('path');
const fs = require('fs');
const modelDir = path.join(__dirname, '..', 'model');
const Ranking = require(path.join(modelDir, 'ranking'));
const Glicko = require(path.join(modelDir, 'glicko'));

const [,, file, evalFromS, evalToS, stepS] = process.argv;
const evalFrom = Number(evalFromS), evalTo = Number(evalToS), step = Number(stepS);

const glicko = new Glicko();
glicko.setFixedRD(75);

// Copied from model/fit.js Fit.priorWinExpectation.
function priorWinExpectation(rv1, rv2) {
    let t1W = glicko.newTeam(rv1), t1L = glicko.newTeam(rv1), t2 = glicko.newTeam(rv2);
    let start = t1W.rank();
    t1W.addPendingMatch(t2, 1); t1L.addPendingMatch(t2, 0);
    t1W.applyPendingMatches(glicko); t1L.applyPendingMatches(glicko);
    let winDelta = t1W.rank() - start, lossDelta = start - t1L.rank();
    return 1 - (winDelta / (winDelta + lossDelta));
}

// fit.js does Math.max(array) here, which is NaN for more than one hit; use the intended max.
function rankValueByRoster(teams, players) {
    let best = -1;
    for (const t of teams)
        if (t.sharesRoster(players) && t.rankValue > best) best = t.rankValue;
    return best;
}

const raw = JSON.parse(fs.readFileSync(file));
const all = raw.matches.filter(m => m.matchStartTime >= evalFrom && m.matchStartTime < evalTo)
    .sort((a, b) => a.matchStartTime - b.matchStartTime);

let i = 0;
for (let T = evalFrom; T < evalTo; T += step) {
    const [, teams] = Ranking.generateRanking(T, file);
    const ranked = teams.filter(t => t.satisfiesRankingCriteria);
    while (i < all.length && all[i].matchStartTime < T + step) {
        const m = all[i++];
        const rv1 = rankValueByRoster(ranked, m.team1Players);
        const rv2 = rankValueByRoster(ranked, m.team2Players);
        const p = (rv1 < 0 || rv2 < 0) ? -1 : priorWinExpectation(rv1, rv2);
        process.stdout.write(`${m.matchStartTime} ${m.team1Id} ${m.team2Id} ${p}\n`);
    }
    process.stderr.write(`ranked ${ranked.length} rosters as of ${new Date(T * 1000).toISOString().slice(0, 10)}; scored through match ${i}/${all.length}\n`);
}
