"""MCP server: lets Claude query the viewer JSON (real pitchers only, held-out games).

    python scripts/mcp_server.py        # stdio; registered for Claude Code in .mcp.json

Tools are thin wrappers over pitchsim.mcp_tools. Synthetic pitchers never appear: the server
only reads web/public/data/, which export_viewer.py refuses to write synthetic data into.
"""
from mcp.server.mcpserver import MCPServer

from pitchsim import mcp_tools as T

mcp = MCPServer("pitchsim")


@mcp.tool()
def list_pitchers() -> list[dict]:
    """Exported pitchers: slug, name, throws, n_at_bats. Use the slug in the other tools."""
    return T.list_pitchers()


@mcp.tool()
def arsenal(slug: str) -> dict:
    """Per pitch type for one pitcher: count, median mph, rpm, approach angles (deg), perceived mph,
    and break vs a no-spin ball (inches). Held-out games only."""
    return T.arsenal(T.load(slug))


@mcp.tool()
def tunnel(slug: str, pitch_a: str, pitch_b: str) -> dict:
    """Median separation (inches) at the hitter's decision point (~25 ft out) between a pitch of
    type pitch_a and the next pitch of type pitch_b in the same at-bat (Statcast codes: FF, SL, CH...).
    Smaller means harder to tell apart early."""
    return T.tunnel(T.load(slug), pitch_a, pitch_b)


@mcp.tool()
def plate_error(slug: str) -> dict:
    """Per pitch type: median miss (inches) between the simulated (M1+M3) and Statcast-fit plate
    position on held-out pitches. How far to trust the simulation for that pitch type."""
    return T.plate_error(T.load(slug))


@mcp.tool()
def compare_pitchers(slug_a: str, slug_b: str) -> dict:
    """Side-by-side arsenal for two pitchers, per pitch type both throw: n, mph, rpm, approach
    angles, perceived mph, break, and simulation miss, each as {a, b, diff = b - a}. Check n first."""
    return T.compare(T.load(slug_a), T.load(slug_b))


@mcp.tool()
def sequence(slug: str, stand: str | None = None, balls: int | None = None,
             strikes: int | None = None) -> dict:
    """What he throws next given the previous pitch type in the at-bat ("start" = first pitch):
    count n, share p, and median tunnel separation (in) per transition. Optional filters: stand
    ("L"/"R" batter) and the balls/strikes the next pitch is thrown in. Unsmoothed; check n."""
    return T.sequence(T.load(slug), stand, balls, strikes)


@mcp.tool()
def generator_check(slug: str, pitch_type: str | None = None) -> list[dict]:
    """M2 check on the per-pitcher generative model: per pitch type and model (gaussian/mixture),
    classifier AUC of real held-out vs generated pitches, the real-vs-real floor (auc_floor), gap,
    margin, and pass (gap <= margin). Not every pitcher has results; graded=false means too few pitches."""
    return T.generator_check(slug, pitch_type)


@mcp.tool()
def sequence_model_check(slug: str) -> dict:
    """M6 held-out log loss per model (lower is better) and calibration gap for pitchers M6
    evaluated (Ohtani, Misiorowski, Sale). Others return evaluated=false: untested, not an error."""
    return T.sequence_model_check(slug)


@mcp.tool()
def next_pitch_probs(slug: str, balls: int, strikes: int, stand: str, prev_type: str = "start",
                     tto: int = 1, sample: bool = False, seed: int = 0) -> dict:
    """P(next pitch type) from the M6 logistic+tto model for a count, batter side (L/R), previous
    pitch type ("start" = first pitch) and times through the order (1-3). sample=true also draws one
    pitch (seeded). Works for any pitcher with cached Statcast data; check sequence_model_check for
    whether M6 validated that pitcher."""
    return T.next_pitch_probs(slug, balls, strikes, stand, prev_type, tto, sample, seed)


@mcp.tool()
def sample_pitches(slug: str, pitch_type: str, n: int = 100, seed: int = 0) -> dict:
    """SYNTHETIC pitches, not real data: draws n releases from the M2 arm for one pitch type, flies
    them with M1 + M3, returns plate position (in, x right / z up), speed, spin, approach angles and
    perceived speed as median/sd, plus 10 example pitches. Check generator_check first for how far
    to trust that pitch type. Slow the first time per pitcher (fits on call)."""
    return T.sample_pitches(slug, pitch_type, n, seed)


@mcp.tool()
def fly_pitch(slug: str, pitch_type: str, release_speed: float | None = None,
              release_spin_rate: float | None = None, spin_axis: float | None = None,
              release_extension: float | None = None, release_pos_x: float | None = None,
              release_pos_z: float | None = None, release_angle_v_deg: float | None = None,
              release_angle_h_deg: float | None = None) -> dict:
    """WHAT-IF flight, not a real pitch: this pitcher's M1+M3 physics for one pitch type. Omitted
    release parameters default to his typical pitch of that type (M2 arm mean), so pass only what you
    want to change, e.g. release_speed=+2 mph faster. Units: mph, rpm, degrees (spin_axis 0-360),
    ft (extension, release x/z). Returns plate position (in), flight time, break vs a no-spin ball
    (in), approach angles and perceived speed."""
    given = dict(release_speed=release_speed, release_spin_rate=release_spin_rate, spin_axis=spin_axis,
                 release_extension=release_extension, release_pos_x=release_pos_x,
                 release_pos_z=release_pos_z, release_angle_v_deg=release_angle_v_deg,
                 release_angle_h_deg=release_angle_h_deg)
    return T.fly_pitch(slug, pitch_type, **{k: v for k, v in given.items() if v is not None})


def _register_doc(name: str) -> None:
    @mcp.resource(f"pitchsim://results/{name}", name=name, mime_type="text/markdown",
                  description=f"results/{name}.md: the milestone write-up (what was measured, the numbers, what is still wrong)")
    def doc() -> str:
        return T.results_doc(name)


for _name in T.results_docs():   # one resource per write-up, so clients can list them
    _register_doc(_name)


if __name__ == "__main__":
    mcp.run()
