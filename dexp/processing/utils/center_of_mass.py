from dexp.processing.backends.backend import Backend


def center_of_mass(image,
                   mode: str = 'projection',
                   projection: str = 'avg',
                   remove_offset: bool = True):

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if mode == 'full':
        if remove_offset:
            image -= image.min()
        com = sp.ndimage.center_of_mass(image)
        com = xp.asarray(com, dtype=xp.float32)
        return com
    elif mode == 'projection':
        ndim = image.ndim

        com = xp.zeros((ndim,), dtype=xp.float32)
        count = xp.zeros((ndim,), dtype=xp.float32)

        for u in range(ndim):
            for v in range(ndim):
                if u < v:
                    proj_axis = tuple(set(range(ndim)).difference({u, v}))
                    if projection == 'avg':
                        projected_image = xp.mean(image, axis=proj_axis)
                    elif projection == 'max':
                        projected_image = xp.max(image, axis=proj_axis)
                    elif projection == 'min':
                        projected_image = xp.min(image, axis=proj_axis)

                    if remove_offset:
                        projected_image -= projected_image.min()

                    du, dv = sp.ndimage.center_of_mass(projected_image)
                    com[u] += du
                    com[v] += dv
                    count[u] += 1
                    count[v] += 1

        com /= count

        return com


