"""Pure helpers for voice-client lifecycle.

Kept side-effect free so the selection logic can be unit-tested without
a live Discord connection (the /disconnect command that uses it can't).
"""


def disconnect_targets(live_vc, helper_vc, voice_clients, guild_id):
    """Ordered, de-duplicated list of voice clients to disconnect for a guild.

    ``helper_vc`` (the BotHelper's cached client) can be *stale* after a
    voice reconnect — disconnecting only it reports success while the bot
    stays in the channel. So we also include the live client
    (``ctx.guild.voice_client``) and any client in ``voice_clients``
    bound to this guild. Order: live first, then helper, then the rest;
    duplicates (by identity/equality) are dropped.
    """
    def _seen(obj):
        # Dedup by identity, not equality: two distinct voice clients
        # must both be disconnected even if some __eq__ deems them equal.
        return any(obj is t for t in targets)

    targets = []
    for cand in (live_vc, helper_vc):
        if cand is not None and not _seen(cand):
            targets.append(cand)
    for vc in voice_clients or []:
        guild = getattr(vc, "guild", None)
        if guild is not None and getattr(guild, "id", None) == guild_id \
                and not _seen(vc):
            targets.append(vc)
    return targets
