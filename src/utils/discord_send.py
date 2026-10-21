"""Sending /ask replies as interaction follow-ups.

Discord makes only the *first* follow-up after a deferred response take
the defer's visibility, because that one replaces the "thinking..."
placeholder. Every later follow-up carries its own flags, and Pycord's
``Webhook.send`` defaults to ``ephemeral=False``. So a private /ask answer
split into several messages would post parts 2..n publicly unless every
send sets the flag (roborev #6486). All /ask follow-ups go through here.
"""


async def send_followups(ctx, parts, *, ephemeral: bool) -> None:
    """Send ``parts`` in order as follow-ups, each with the same
    visibility. Raises whatever the first failing send raises."""
    for part in parts:
        await ctx.followup.send(part, ephemeral=ephemeral)
