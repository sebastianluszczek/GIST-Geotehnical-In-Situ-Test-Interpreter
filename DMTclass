import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DMT(object):
    'Klasa operująca na danych pomiarowych DMT'

    def __init__(self, plik, dA, dB, wlvl):
        df = pd.read_csv(plik, delimiter=',', decimal=',')
        self.df = df.fillna(0.0)
        self.dA = dA
        self.dB = dB
        self.wlvl = wlvl
        print('DataFrame "{}" został zaimportowany do klasy DMT\n'.format(plik))

    def pokaz(self, wiersze=10):
        print(self.df.head(wiersze), '\n')

    def interpretacja(self):

        self.df['p0']=1.05*(self.df['A ']+self.dA)-0.05*(self.df['B']-self.dB)
        self.df['p1']=self.df['B']-self.dB

        self.df['ED'] = 34.7*(self.df['p1']-self.df['p0'])

        wlvl_mask = self.df['Depth (m)'] >= self.wlvl
        self.df['u0'] = (self.df['Depth (m)'] - self.wlvl) * 0.098
        self.df['u0'] = self.df['u0'][wlvl_mask]
        self.df = self.df.fillna(0.0)

        self.df['ID'] = (self.df['p1'] - self.df['p0'])/(self.df['p0'] - self.df['u0'])

        self.df['ED'] = 34.7 * (self.df['p1'] - self.df['p0'])

        #'funkcja określająca ciężar wlasciwy [...]'
        Am, An, Bm, Bn, Cm, Cn, Dm, Dn = 0.585, 1.737, 0.621, 2.013, 0.657, 2.289, 0.694, 2.564

        def uw(ID,ED):
            if ID < 0.6:
                if ED >= 10 ** (Dn + Dm * np.log10(ID)):
                    return 2.05
                elif ED >= 10 ** (Cn + Cm * np.log10(ID)):
                    return 1.9
                elif ED >= 10 ** (Bn + Bm * np.log10(ID)):
                    return 1.8
                elif ED >= 10 ** (An + Am * np.log10(ID)):
                    return 1.7
                else:
                    return 1.6
            elif ID < 1.8:
                if ED >= 10 ** (Dn + Dm * np.log10(ID)):
                    return 2.1
                elif ED >= 10 ** (Cn + Cm * np.log10(ID)):
                    return 1.95
                elif ED >= 10 ** (Bn + Bm * np.log10(ID)):
                    return 1.8
                elif ED >= 10 ** (An + Am * np.log10(ID)):
                    return 1.7
                else:
                    return 1.6
            else:
                if ED >= 10 ** (Dn + Dm * np.log10(ID)):
                    return 2.15
                elif ED >= 10 ** (Cn + Cm * np.log10(ID)):
                    return 2.0
                elif ED >= 10 ** (Bn + Bm * np.log10(ID)):
                    return 1.9
                elif ED >= 10 ** (An + Am * np.log10(ID)):
                    return 1.8
                else:
                    return 1.7

        self.df['gamma'] = self.df.apply(lambda x: uw(x['ID'], x['ED']), axis = 1)

        self.df['sigma'] = self.df['gamma'].cumsum()*0.02
        self.df['sigma_v0'] = self.df['gamma'].cumsum() * 0.02 - self.df['u0']

        self.df['KD'] = (self.df['p0'] - self.df['u0']) / self.df['sigma_v0']

        def description(ID, ED):
            if ED >= 12:
                if ID <= 0.33:
                    return "CLAY"
                elif ID <= 0.66:
                    return "SILTY CLAY"
                elif ID <= 0.8:
                    return "CLAYEY SILT"
                elif ID <= 1.2:
                    return "SILT"
                elif ID <= 1.8:
                    return "SANDY SILT"
                elif ID <= 3.3:
                    return "SILTY SAND"
                else:
                    return "SAND"
            else:
                return "MUD"

        self.df['description'] = self.df.apply(lambda x: description(x['ID'], x['ED']), axis = 1)

        def coeff_earth_preasure(ID, KD):
            if ID < 1.2:
                return ((KD / 1.5) ** 0.47) - 0.6
            else:
                return 0
        self.df['K0'] = self.df.apply(lambda x: coeff_earth_preasure(x['ID'], x['ED']), axis = 1)

        def overconsolidation_ratio(ID, KD):
            if ID < 1.2:
                return (0.5 * KD) ** 1.56
            else:
                return 0

        self.df['OCR'] = self.df.apply(lambda x: overconsolidation_ratio(x['ID'], x['KD']), axis = 1)

        def undrained_shear_strenght(ID, sigma_v0, KD):
            if ID < 1.2:
                return 0.22 * sigma_v0 * (0.5 * KD) ** 1.25
            else:
                return 0

        self.df['cu'] = self.df.apply(lambda x: undrained_shear_strenght(x['ID'],x['sigma_v0'], x['KD']), axis=1)

        def friction_angle(ID, KD):
            if ID > 1.2:
                return 28 + 14.6 * np.log10(KD) - 2.1 * (np.log10(KD)) ** 2
            else:
                return 0

        self.df['phi'] = self.df.apply(lambda x: friction_angle(x['ID'], x['KD']), axis=1)

        def M_const_modulus(ID, KD, ED):
            if KD <= 10:
                if ID <= 0.6:
                    RM = 0.14 + 2.36 * np.log10(KD)
                    if RM >= 0.85:
                        return ED * RM
                    else:
                        RM = 0.85
                        return ED * RM
                elif ID < 3.0:
                    RM0 = 0.14 + 0.15 * (ID - 0.6)
                    RM = RM0 + (2.5 - RM0) * np.log(KD)
                    if RM >= 0.85:
                        return ED * RM
                    else:
                        RM = 0.85
                        return ED * RM
                else:
                    RM = 0.5 + 2.0 * np.log(KD)
                    if RM >= 0.85:
                        return RM * ED
                    else:
                        RM = 0.85
                        return ED * RM
            else:
                RM = 0.32 + 2.18 * np.log10(KD)
                return RM * ED

        self.df['M'] = self.df.apply(lambda x: M_const_modulus(x['ID'], x['KD'], x['ED']), axis=1)

        return self.df

    def sum(self, col):
        print(self.df[col].sum(axis = 0))

    def eksport(self):
        self.df.to_csv(path_or_buf='DMT_int_wynik.csv')

    def wykres(self, k1, k2, k3):
        ylim = (self.df['Depth (m)'].max(), self.df['Depth (m)'].min())

        f = plt.figure(figsize=(15, 10))
        ax1 = f.add_subplot(131)
        ax2 = f.add_subplot(132)
        ax3 = f.add_subplot(133)

        ax1.plot(self.df[k1], self.df['Depth (m)'], 'k-', linewidth=1.0)
        ax1.set_ylim(ylim)
        ax1.set(xlabel=k1, ylabel='Depth (m)')
        ax1.grid(True)

        ax2.plot(self.df[k2], self.df['Depth (m)'], 'k-', linewidth=1.0)
        ax2.set_ylim(ylim)
        ax2.set(xlabel=k2)
        ax2.grid(True)

        ax3.plot(self.df[k3], self.df['Depth (m)'], 'k-', linewidth=1.0)
        ax3.set_ylim(ylim)
        ax3.set(xlabel=k3)
        ax3.grid(True)

        plt.show()


