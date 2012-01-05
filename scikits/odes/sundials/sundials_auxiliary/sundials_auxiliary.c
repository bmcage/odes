#include <sundials/sundials_types.h>
#include <sundials/sundials_direct.h>
#include <sundials/sundials_nvector.h>
#include <nvector/nvector_serial.h>

/* N_Vector content access functions */
inline N_VectorContent_Serial nv_content_s(N_Vector v) {
    return (N_VectorContent_Serial)(v->content);
}

inline long int nv_length_s(N_VectorContent_Serial vc_s) {
    return vc_s->length;
}

inline booleantype nv_own_data_s(N_VectorContent_Serial vc_s) {
    return vc_s->own_data;
}

inline realtype* nv_data_s(N_VectorContent_Serial vc_s) {
    return vc_s->data;
}

typedef realtype *nv_content_data_s;

inline realtype get_nv_ith_s(nv_content_data_s vcd_s, int i) {
    return vcd_s[i];
}

inline void set_nv_ith_s(nv_content_data_s vcd_s, int i,
                          realtype new_value) {
    vcd_s[i] = new_value;
}

/* Dense matrix: acces functions */
typedef realtype *DlsMat_col;

inline int get_dense_N(DlsMat A) {
    return A->N;
}

inline int get_dense_M(DlsMat A) {
    return A->M;
}

inline int get_band_mu(DlsMat A) {
    return A->mu;
}

inline int get_band_ml(DlsMat A) {
    return A->ml;
}
inline realtype* get_dense_col(DlsMat A, int j) {
    return (A->cols)[j];
}
    
inline void set_dense_col(DlsMat A, int j, realtype *data) {
    (A->cols)[j] = data;
}
    
inline realtype get_dense_element(DlsMat A, int i, int j) {
    return (A->cols)[j][i];
}
    
inline void set_dense_element(DlsMat A, int i, int j, realtype aij) {
    (A->cols)[j][i] = aij;
}

/* Band matrix acces functions */
inline DlsMat_col get_band_col(DlsMat A, int j) {
    return ((A->cols)[j] + (A->s_mu));
}
    
inline void set_band_col(DlsMat A, int j, realtype *data) {
    ((A->cols)[j]) = data;
}

inline realtype get_band_col_elem(DlsMat_col col_j, int i, int j) {
    return col_j[(i)-(j)];
}
    
inline void set_band_col_elem(DlsMat_col col_j, int i, int j, realtype aij) {
    col_j[(i)-(j)] = aij;
}
inline realtype get_band_element(DlsMat A, int i, int j) {
    return ((A->cols)[j][(i)-(j)+(A->s_mu)]);
}
    
inline void set_band_element(DlsMat A, int i, int j, realtype aij) {
    (A->cols)[j][(i)-(j)+(A->s_mu)] = aij;
}

