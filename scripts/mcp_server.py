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


if __name__ == "__main__":
    mcp.run()
