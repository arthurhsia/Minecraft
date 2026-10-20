"""Focus gating: nothing is sent unless the game is the frontmost app."""
from mcbio.outputs import DryRunOutput, FocusGatedOutput
from mcbio.types import Commands


def test_gate_blocks_and_releases():
    front = {"name": "Finder"}
    t = {"now": 0.0}
    inner = DryRunOutput()
    g = FocusGatedOutput(inner, probe=lambda: front["name"], clock=lambda: t["now"], check_every_s=0.0)
    walk = Commands(keys={"forward"}, mouse_dx=5, left_button=True)
    g.apply(walk)
    assert inner.events == [] and not g.focused                     # Finder in front: nothing sent
    front["name"] = "Minecraft Education"; t["now"] = 1
    g.apply(walk)
    assert g.focused and inner.held_keys == {"forward"} and "L" in inner.held_buttons
    front["name"] = "Safari"; t["now"] = 2
    n = len(inner.events)
    g.apply(walk)
    assert not g.focused and inner.held_keys == set() and inner.held_buttons == set()
    assert all(e[0] != "move" for e in inner.events[n:])            # released, no new motion
