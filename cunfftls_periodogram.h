/* cunfftls_periodogram.h
 * ======================
 * 
 * Implementation of Lomb-Scargle periodogram
 *
 * (c) 2016, John Hoffman 
 * using code borrowed extensively from B. Leroy's nfftls
 *
 * This file is part of cunfftls.
 *
 * cunfftls is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cunfftls is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cunfftls.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright (C) 2012 by B. Leroy [nfftls], 2016 by J. Hoffman [cunfftls]
 */
#ifndef CUNFFTLS_PERIODOGRAM_H
#define CUNFFTLS_PERIODOGRAM_H
#include "cunfftls_typedefs.h"

__host__  dTyp *
lombScargle(const dTyp *tobs, const dTyp *yobs, int npts, 
            Settings *settings); 

__host__  dTyp *
lombScargleBatch(const dTyp *tobs, const dTyp *yobs, int *npts, int nlc, 
            Settings *settings); 

__host__ dTyp *
generalizedLombScargle(const dTyp *obs, const dTyp *yobs, const dTyp *errs,
            int npts, Settings *settings);

__host__ dTyp 
probability(dTyp Pn, int npts, int nfreqs, dTyp over);

__host__ dTyp 
logProba(dTyp Pn, int npts, int nfreqs, dTyp over);

__host__ dTyp getPnCutoffBootstrap(dTyp proba, dTyp mu, dTyp sig);

__host__ dTyp getPnCutoff(dTyp proba, int npts, int nfreqs, dTyp over);
#endif
