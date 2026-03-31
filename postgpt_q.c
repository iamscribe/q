/*
 * postgpt_q.c — PostGPT-Q: Resonant Reasoning Engine (C inference)
 *
 * Triple attention: Content (QK^T) + RRPRAM (x@Wr) + Janus echo (W^T·W)
 * Learned gating between mechanisms.
 * Dario equation: bigram + trigram + hebbian + destiny.
 * Transformer gate: untrained = silent, trained = speaks.
 * 6 Kuramoto chambers. Calendar drift. 12 bidirectional steps.
 *
 * cc postgpt_q.c -O2 -lm -o q && ./q weights.bin q.merges q.txt
 *
 * (c) 2026 arianna method
 * resonance is unbreakable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define MAX_VOCAB    1280
#define MAX_CTX      128
#define MAX_BPE      1024
#define MAX_SEQ      4096
#define MAX_BIGRAM   65536
#define MAX_TRIGRAM  65536
#define MAX_HEBBIAN  131072
#define N_CHAMBERS   6
#define CHAIN_STEPS  12
#define TOP_K        15
#define QPTQ_MAGIC   0x51505451

/* ── math ── */
static float clampf(float x, float lo, float hi) { return x<lo?lo:x>hi?hi:x; }

static void rmsnorm(float *out, const float *x, int n) {
    float ms=0; for(int i=0;i<n;i++) ms+=x[i]*x[i];
    ms=1.0f/sqrtf(ms/n+1e-6f);
    for(int i=0;i<n;i++) out[i]=x[i]*ms;
}

/* w stored as [d_out, n_in] row-major (PyTorch nn.Linear convention) */
static void matmul(float *out, const float *x, const float *w, int n_in, int d_out) {
    for(int d=0;d<d_out;d++){
        float v=0; for(int j=0;j<n_in;j++) v+=x[j]*w[d*n_in+j];
        out[d]=v;
    }
}

static void softmax(float *x, int n) {
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}
    if(s>0) for(int i=0;i<n;i++) x[i]/=s;
}

static int sample_nucleus(float *logits, int V, float temp, float top_p) {
    /* top-p (nucleus) sampling: sort by prob, sample from smallest set summing to p */
    int idx[TOP_K]; float val[TOP_K];
    for(int k=0;k<TOP_K;k++){idx[k]=0;val[k]=-1e30f;}
    for(int i=0;i<V;i++){
        if(logits[i]>val[TOP_K-1]){
            val[TOP_K-1]=logits[i];idx[TOP_K-1]=i;
            for(int k=TOP_K-2;k>=0;k--){
                if(val[k+1]>val[k]){float tv=val[k];val[k]=val[k+1];val[k+1]=tv;
                    int ti=idx[k];idx[k]=idx[k+1];idx[k+1]=ti;}else break;}
        }
    }
    float mx=val[0],pr[TOP_K],tot=0;
    for(int k=0;k<TOP_K;k++){pr[k]=expf((val[k]-mx)/temp);tot+=pr[k];}
    /* nucleus: find smallest k such that sum of top-k probs >= top_p */
    float cum=0; int nk=TOP_K;
    for(int k=0;k<TOP_K;k++){cum+=pr[k]/tot;if(cum>=top_p){nk=k+1;break;}}
    /* resample from nucleus */
    float ntot=0;for(int k=0;k<nk;k++) ntot+=pr[k];
    float r=(float)rand()/RAND_MAX*ntot; cum=0;
    for(int k=0;k<nk;k++){cum+=pr[k];if(cum>=r)return idx[k];}
    return idx[0];
}

/* ── BPE ── */
typedef struct{int a,b,new_id;}BPEMerge;
typedef struct{
    BPEMerge merges[MAX_BPE]; int n_merges,vocab_size;
    uint8_t vocab_bytes[MAX_VOCAB][64]; int vocab_len[MAX_VOCAB];
}BPE;

static int bpe_load(BPE *bpe, const char *path){
    FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"ERROR: %s\n",path);return 0;}
    uint32_t n; fread(&n,4,1,f); bpe->n_merges=(int)n; bpe->vocab_size=256+n;
    for(int i=0;i<256;i++){bpe->vocab_bytes[i][0]=(uint8_t)i;bpe->vocab_len[i]=1;}
    for(int i=0;i<(int)n&&i<MAX_BPE;i++){
        uint32_t a,b,nid; fread(&a,4,1,f);fread(&b,4,1,f);fread(&nid,4,1,f);
        bpe->merges[i].a=a;bpe->merges[i].b=b;bpe->merges[i].new_id=nid;
        int la=bpe->vocab_len[a],lb=bpe->vocab_len[b];
        if(la+lb<64){memcpy(bpe->vocab_bytes[nid],bpe->vocab_bytes[a],la);
            memcpy(bpe->vocab_bytes[nid]+la,bpe->vocab_bytes[b],lb);
            bpe->vocab_len[nid]=la+lb;}
    }
    fclose(f); return 1;
}

static int bpe_encode(const BPE *bpe, const uint8_t *text, int tlen, int *out, int maxo){
    int n=0; for(int i=0;i<tlen&&n<maxo;i++) out[n++]=text[i];
    for(int m=0;m<bpe->n_merges;m++){
        int a=bpe->merges[m].a,b=bpe->merges[m].b,nid=bpe->merges[m].new_id,j=0;
        for(int i=0;i<n;i++){if(i<n-1&&out[i]==a&&out[i+1]==b){out[j++]=nid;i++;}else out[j++]=out[i];}
        n=j;
    }
    return n;
}

static int bpe_decode_token(const BPE *bpe, int id, char *buf, int sz){
    if(id<0||id>=bpe->vocab_size)return 0;
    int len=bpe->vocab_len[id]; if(len>=sz)len=sz-1;
    memcpy(buf,bpe->vocab_bytes[id],len); buf[len]=0; return len;
}

/* ── MetaWeights ── */
typedef struct{int a,b;float prob;}BigramE;
typedef struct{int a,b,c;float prob;}TrigramE;
typedef struct{int a,b;float str;}HebbE;
typedef struct{
    float unigram[MAX_VOCAB];
    BigramE bigrams[MAX_BIGRAM]; int n_bi;
    TrigramE trigrams[MAX_TRIGRAM]; int n_tri;
    HebbE hebbs[MAX_HEBBIAN]; int n_hebb;
}MetaW;

static void meta_build(MetaW *mw, const int *ids, int n, int V){
    memset(mw,0,sizeof(*mw));
    for(int i=0;i<n;i++) if(ids[i]<V) mw->unigram[ids[i]]+=1.0f;
    float tot=0; for(int i=0;i<V;i++) tot+=mw->unigram[i];
    if(tot>0) for(int i=0;i<V;i++) mw->unigram[i]/=tot;
    /* bigram */
    typedef struct{int a,b;float c;}BC;
    BC *bc=calloc(MAX_BIGRAM,sizeof(BC)); int nbc=0;
    for(int i=0;i<n-1&&nbc<MAX_BIGRAM-1;i++){
        int a=ids[i],b=ids[i+1]; int found=0;
        for(int j=0;j<nbc;j++) if(bc[j].a==a&&bc[j].b==b){bc[j].c+=1;found=1;break;}
        if(!found){bc[nbc].a=a;bc[nbc].b=b;bc[nbc].c=1;nbc++;}
    }
    for(int i=0;i<nbc;i++){
        float t=0; for(int j=0;j<nbc;j++) if(bc[j].a==bc[i].a) t+=bc[j].c;
        if(t>0){mw->bigrams[mw->n_bi].a=bc[i].a;mw->bigrams[mw->n_bi].b=bc[i].b;
            mw->bigrams[mw->n_bi].prob=bc[i].c/t;mw->n_bi++;}
    }
    free(bc);
    /* trigram */
    typedef struct{int a,b,c;float cnt;}TC;
    TC *tc=calloc(MAX_TRIGRAM,sizeof(TC)); int ntc=0;
    for(int i=0;i<n-2&&ntc<MAX_TRIGRAM-1;i++){
        int a=ids[i],b=ids[i+1],c=ids[i+2]; int found=0;
        for(int j=0;j<ntc;j++) if(tc[j].a==a&&tc[j].b==b&&tc[j].c==c){tc[j].cnt+=1;found=1;break;}
        if(!found){tc[ntc].a=a;tc[ntc].b=b;tc[ntc].c=c;tc[ntc].cnt=1;ntc++;}
    }
    for(int i=0;i<ntc&&mw->n_tri<MAX_TRIGRAM;i++){
        float t=0; for(int j=0;j<ntc;j++) if(tc[j].a==tc[i].a&&tc[j].b==tc[i].b) t+=tc[j].cnt;
        if(t>0){mw->trigrams[mw->n_tri].a=tc[i].a;mw->trigrams[mw->n_tri].b=tc[i].b;
            mw->trigrams[mw->n_tri].c=tc[i].c;mw->trigrams[mw->n_tri].prob=tc[i].cnt/t;mw->n_tri++;}
    }
    free(tc);
    /* hebbian */
    int hn=n<8000?n:8000, win=5;
    for(int i=0;i<hn&&mw->n_hebb<MAX_HEBBIAN-1;i++){
        for(int j=(i-win>0?i-win:0);j<hn&&j<=i+win;j++){
            if(i==j)continue;
            int a=ids[i]<ids[j]?ids[i]:ids[j],b=ids[i]<ids[j]?ids[j]:ids[i];
            float decay=1.0f/(1.0f+abs(i-j)); int found=0;
            for(int k=0;k<mw->n_hebb;k++) if(mw->hebbs[k].a==a&&mw->hebbs[k].b==b){mw->hebbs[k].str+=decay;found=1;break;}
            if(!found&&mw->n_hebb<MAX_HEBBIAN-1){mw->hebbs[mw->n_hebb].a=a;mw->hebbs[mw->n_hebb].b=b;mw->hebbs[mw->n_hebb].str=decay;mw->n_hebb++;}
        }
    }
    float mx=0; for(int i=0;i<mw->n_hebb;i++) if(mw->hebbs[i].str>mx) mx=mw->hebbs[i].str;
    if(mx>0) for(int i=0;i<mw->n_hebb;i++) mw->hebbs[i].str/=mx;
    printf("  metaweights: %d bi, %d tri, %d hebb\n",mw->n_bi,mw->n_tri,mw->n_hebb);
}

static float meta_bi(const MetaW *mw, int prev, int next){
    for(int i=0;i<mw->n_bi;i++) if(mw->bigrams[i].a==prev&&mw->bigrams[i].b==next) return mw->bigrams[i].prob;
    return 1e-10f;
}
static float meta_tri(const MetaW *mw, int p2, int p1, int next){
    for(int i=0;i<mw->n_tri;i++) if(mw->trigrams[i].a==p2&&mw->trigrams[i].b==p1&&mw->trigrams[i].c==next) return mw->trigrams[i].prob;
    return 1e-10f;
}
static void meta_hebb(const MetaW *mw, const int *ctx, int cl, float *out, int V){
    memset(out,0,V*sizeof(float));
    for(int ci=0;ci<cl;ci++){int c=ctx[ci];
        for(int k=0;k<mw->n_hebb;k++){
            if(mw->hebbs[k].a==c&&mw->hebbs[k].b<V) out[mw->hebbs[k].b]+=mw->hebbs[k].str;
            else if(mw->hebbs[k].b==c&&mw->hebbs[k].a<V) out[mw->hebbs[k].a]+=mw->hebbs[k].str;
        }
    }
    float mx=0; for(int i=0;i<V;i++) if(out[i]>mx) mx=out[i];
    if(mx>0) for(int i=0;i<V;i++) out[i]/=mx;
}
/* prophecy: predict next token from recent bigram context (top-16) */
static void meta_prophecy(const MetaW *mw, const int *ctx, int cl, float *out, int V){
    memset(out,0,V*sizeof(float));
    int appeared[256]={0}; int na=cl<256?cl:256;
    for(int i=cl-na;i<cl;i++) if(ctx[i]<256) appeared[ctx[i]]=1;
    int start=cl>4?cl-4:0;
    for(int ci=start;ci<cl;ci++){
        int c=ctx[ci];
        for(int k=0;k<mw->n_bi;k++){
            if(mw->bigrams[k].a==c&&mw->bigrams[k].b<V&&!appeared[mw->bigrams[k].b%256]){
                out[mw->bigrams[k].b]+=mw->bigrams[k].prob;
            }
        }
    }
    float mx=0; for(int i=0;i<V;i++) if(out[i]>mx) mx=out[i];
    if(mx>0) for(int i=0;i<V;i++) out[i]/=mx;
}

/* ── Chambers ── */
enum{CH_FEAR=0,CH_LOVE,CH_RAGE,CH_VOID,CH_FLOW,CH_CMPLX};
static const char *CH_N[]={"FEAR","LOVE","RAGE","VOID","FLOW","CMPLX"};
static const float CH_D[]={0.90f,0.93f,0.85f,0.97f,0.88f,0.94f};
static const float COU[6][6]={
    {0,-0.3f,0.5f,0.4f,-0.2f,0.1f},{-0.3f,0,-0.4f,-0.5f,0.5f,0.2f},
    {0.5f,-0.3f,0,0.2f,-0.3f,0.3f},{0.4f,-0.5f,0.3f,0,-0.3f,0.4f},
    {-0.2f,0.4f,-0.2f,-0.3f,0,0.3f},{0.1f,0.2f,0.3f,0.4f,0.3f,0}};
typedef struct{float act[6];float debt;float trauma;}Chambers;

static void ch_init(Chambers *c){memset(c,0,sizeof(*c));c->act[CH_LOVE]=0.2f;c->act[CH_FLOW]=0.15f;c->trauma=0;}
static void ch_xfire(Chambers *c, int it){
    for(int t=0;t<it;t++){float old[6];memcpy(old,c->act,sizeof(old));
        for(int i=0;i<6;i++){c->act[i]*=CH_D[i];
            for(int j=0;j<6;j++) if(i!=j) c->act[i]+=0.03f*COU[i][j]*sinf(old[j]-old[i]);
            c->act[i]=clampf(c->act[i],0,1);}
    }
}

/* ── DOE Parliament — Democracy of Experts ── */
/* LoRA experts that vote, split (mitosis), die (apoptosis).
   NOTORCH: Hebbian update per forward pass, no backward.
   θ = ε + γ + αδ  where δ = parliament injection */

#define MAX_EXPERTS  16
#define DOE_RANK     4
#define DOE_ALPHA    0.05f

typedef struct{
    float *A;  /* [rank × d_in] */
    float *B;  /* [d_out × rank] */
    int d_in,d_out,rank;
    float vitality;
    int age,low_steps;
}Expert;

typedef struct{
    Expert ex[MAX_EXPERTS]; int n;
    int d_model; float alpha;
    int step;
}Parliament;

static void expert_init(Expert *e, int d_in, int d_out, int rank){
    e->d_in=d_in;e->d_out=d_out;e->rank=rank;
    e->A=calloc(rank*d_in,sizeof(float));
    e->B=calloc(d_out*rank,sizeof(float));
    for(int i=0;i<rank*d_in;i++) e->A[i]=0.01f*((float)rand()/RAND_MAX-0.5f);
    for(int i=0;i<d_out*rank;i++) e->B[i]=0.01f*((float)rand()/RAND_MAX-0.5f);
    e->vitality=1.0f;e->age=0;e->low_steps=0;
}

static void expert_forward(const Expert *e, const float *x, float *out){
    /* out[d_out] = B @ (A @ x) */
    float mid[DOE_RANK];
    for(int r=0;r<e->rank;r++){float s=0;for(int d=0;d<e->d_in;d++) s+=e->A[r*e->d_in+d]*x[d];mid[r]=s;}
    for(int o=0;o<e->d_out;o++){float s=0;for(int r=0;r<e->rank;r++) s+=e->B[o*e->rank+r]*mid[r];out[o]=s;}
}

static void expert_hebbian(Expert *e, const float *x, const float *dy, float lr){
    /* NOTORCH: Hebbian update. dy = prophecy debt signal */
    for(int r=0;r<e->rank;r++){
        float u=0;for(int o=0;o<e->d_out;o++) u+=e->B[o*e->rank+r]*dy[o];
        u+=0.01f*((float)rand()/RAND_MAX-0.5f);
        for(int d=0;d<e->d_in;d++) e->A[r*e->d_in+d]+=lr*x[d]*u;
        for(int o=0;o<e->d_out;o++) e->B[o*e->rank+r]*=0.999f;
    }
}

static void parl_init(Parliament *p, int d_model, int n_init){
    p->d_model=d_model;p->alpha=DOE_ALPHA;p->step=0;
    p->n=n_init<MAX_EXPERTS?n_init:MAX_EXPERTS;
    for(int i=0;i<p->n;i++) expert_init(&p->ex[i],d_model,d_model,DOE_RANK);
}

static void parl_election(Parliament *p, const float *x, float *result){
    /* Variable-k election. Consensus determines how many experts vote. */
    memset(result,0,p->d_model*sizeof(float));
    if(p->n==0) return;
    float votes[MAX_EXPERTS],*outs[MAX_EXPERTS];
    for(int i=0;i<p->n;i++){
        outs[i]=calloc(p->d_model,sizeof(float));
        expert_forward(&p->ex[i],x,outs[i]);
        float dot=0;for(int d=0;d<p->d_model;d++) dot+=outs[i][d]*x[d];
        votes[i]=dot;
    }
    /* consensus */
    float mx=votes[0],mn=votes[0];
    for(int i=1;i<p->n;i++){if(votes[i]>mx)mx=votes[i];if(votes[i]<mn)mn=votes[i];}
    float cons=(mx-mn)/(fabsf(mx)+fabsf(mn)+1e-8f);
    int k=(int)(p->n*(1.0f-cons));if(k<1)k=1;if(k>p->n)k=p->n;
    /* top-k by insertion sort on indices */
    int sel[MAX_EXPERTS];for(int i=0;i<p->n;i++) sel[i]=i;
    for(int i=0;i<p->n-1;i++) for(int j=i+1;j<p->n;j++)
        if(votes[sel[j]]>votes[sel[i]]){int t=sel[i];sel[i]=sel[j];sel[j]=t;}
    /* softmax over top-k */
    float sv=votes[sel[0]],exps[MAX_EXPERTS],tot=0;
    for(int i=0;i<k;i++){exps[i]=expf(votes[sel[i]]-sv);tot+=exps[i];}
    for(int i=0;i<k;i++){
        float w=exps[i]/tot;
        for(int d=0;d<p->d_model;d++) result[d]+=w*outs[sel[i]][d];
        p->ex[sel[i]].vitality=0.9f*p->ex[sel[i]].vitality+0.1f*fabsf(w);
    }
    for(int i=k;i<p->n;i++){p->ex[sel[i]].vitality*=0.95f;p->ex[sel[i]].low_steps++;}
    for(int i=0;i<p->n;i++) free(outs[i]);
}

static void parl_inject(Parliament *p, float *logits, const float *x, int V){
    float *delta=calloc(p->d_model,sizeof(float));
    parl_election(p,x,delta);
    int n=V<p->d_model?V:p->d_model;
    for(int i=0;i<n;i++) logits[i]+=p->alpha*delta[i];
    free(delta);
}

static void parl_notorch(Parliament *p, const float *x, const float *debt, int dlen){
    int n=dlen<p->d_model?dlen:p->d_model;
    float *ds=calloc(p->d_model,sizeof(float));
    for(int i=0;i<n;i++) ds[i]=debt[i];
    for(int i=0;i<p->n;i++){expert_hebbian(&p->ex[i],x,ds,0.001f);p->ex[i].age++;}
    free(ds);
}

static void parl_lifecycle(Parliament *p){
    /* apoptosis */
    int alive=0;
    for(int i=0;i<p->n;i++){
        if(p->ex[i].low_steps>=8&&p->ex[i].vitality<0.1f&&p->n>2){
            free(p->ex[i].A);free(p->ex[i].B);continue;}
        if(alive!=i) p->ex[alive]=p->ex[i];alive++;
    }
    p->n=alive;
    /* mitosis */
    int births=0;
    for(int i=0;i<p->n&&p->n+births<MAX_EXPERTS;i++){
        if(p->ex[i].vitality>0.8f&&p->ex[i].age>50){
            Expert *c=&p->ex[p->n+births];
            expert_init(c,p->ex[i].d_in,p->ex[i].d_out,p->ex[i].rank);
            for(int j=0;j<c->rank*c->d_in;j++) c->A[j]=p->ex[i].A[j]+0.005f*((float)rand()/RAND_MAX-0.5f);
            for(int j=0;j<c->d_out*c->rank;j++) c->B[j]=p->ex[i].B[j]+0.005f*((float)rand()/RAND_MAX-0.5f);
            c->vitality=0.5f;births++;
            p->ex[i].vitality*=0.6f;
        }
    }
    p->n+=births;p->step++;
}

/* ── Transformer ── */
typedef struct{
    int V,D,NH,NL,CTX,NC,NR,NJ,HD;
    float *tok,*pos;
    struct{float *wq,*wk,*vc,*wr,*vr,*wj,*vj,*gw,*gb,*wo,*up,*dn;}*L;
    float **kc,**vcc,**vrc; int clen;
    float *logits;
}TF;

static int tf_load(TF *t, const char *path){
    FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"ERROR: %s\n",path);return 0;}
    uint32_t magic; fread(&magic,4,1,f);
    if(magic!=QPTQ_MAGIC){fprintf(stderr,"bad magic\n");fclose(f);return 0;}
    uint32_t ver,v,d,nh,nl,ctx,nc,nr,nj,hd;
    fread(&ver,4,1,f);fread(&v,4,1,f);fread(&d,4,1,f);fread(&nh,4,1,f);
    fread(&nl,4,1,f);fread(&ctx,4,1,f);fread(&nc,4,1,f);fread(&nr,4,1,f);
    fread(&nj,4,1,f);fread(&hd,4,1,f);
    t->V=v;t->D=d;t->NH=nh;t->NL=nl;t->CTX=ctx;t->NC=nc;t->NR=nr;t->NJ=nj;t->HD=hd;
    int nm=(nc>0)+(nr>0)+(nj>0);
    printf("  model: V=%d D=%d H=%d L=%d nc=%d nr=%d nj=%d\n",v,d,nh,nl,nc,nr,nj);
    #define AR(ptr,cnt) do{(ptr)=calloc((cnt),sizeof(float));fread((ptr),sizeof(float),(cnt),f);}while(0)
    AR(t->tok,v*d); AR(t->pos,ctx*d);
    t->L=calloc(nl,sizeof(t->L[0]));
    for(int li=0;li<(int)nl;li++){
        if(nc>0){AR(t->L[li].wq,nc*hd*d);AR(t->L[li].wk,nc*hd*d);AR(t->L[li].vc,nc*hd*d);}
        if(nr>0){AR(t->L[li].wr,nr*d*ctx);AR(t->L[li].vr,nr*hd*d);}
        if(nj>0){AR(t->L[li].wj,nj*hd*d);AR(t->L[li].vj,nj*hd*d);}
        if(nm>1){AR(t->L[li].gw,nm*d);AR(t->L[li].gb,nm);}
        AR(t->L[li].wo,d*d);AR(t->L[li].up,4*d*d);AR(t->L[li].dn,d*4*d);
    }
    #undef AR
    t->kc=calloc(nl,sizeof(float*));t->vcc=calloc(nl,sizeof(float*));t->vrc=calloc(nl,sizeof(float*));
    for(int li=0;li<(int)nl;li++){
        t->kc[li]=calloc(ctx*(nc>0?nc*hd:1),sizeof(float));
        t->vcc[li]=calloc(ctx*(nc>0?nc*hd:1),sizeof(float));
        t->vrc[li]=calloc(ctx*(nr>0?nr*hd:1),sizeof(float));
    }
    t->clen=0; t->logits=calloc(v,sizeof(float));
    fclose(f); return 1;
}

static void tf_reset(TF *t){t->clen=0;}

static void tf_forward(TF *t, int tok, int pos){
    int D=t->D,HD=t->HD,NC=t->NC,NR=t->NR,NJ=t->NJ;
    int nm=(NC>0)+(NR>0)+(NJ>0), sl=pos+1;
    float *x=calloc(D,sizeof(float)),*xn=calloc(D,sizeof(float)),*xr=calloc(D,sizeof(float));
    for(int d=0;d<D;d++) x[d]=t->tok[tok*D+d]+t->pos[pos*D+d];
    for(int li=0;li<t->NL;li++){
        memcpy(xr,x,D*sizeof(float)); rmsnorm(xn,x,D);
        float *co=NULL,*ro=NULL,*jo=NULL;
        /* content */
        if(NC>0){co=calloc(NC*HD,sizeof(float));
            float *q=calloc(NC*HD,sizeof(float)),*k=calloc(NC*HD,sizeof(float)),*vc=calloc(NC*HD,sizeof(float));
            matmul(q,xn,t->L[li].wq,D,NC*HD);matmul(k,xn,t->L[li].wk,D,NC*HD);matmul(vc,xn,t->L[li].vc,D,NC*HD);
            memcpy(t->kc[li]+pos*NC*HD,k,NC*HD*sizeof(float));
            memcpy(t->vcc[li]+pos*NC*HD,vc,NC*HD*sizeof(float));
            for(int h=0;h<NC;h++){
                float *sc=calloc(sl,sizeof(float));
                for(int p=0;p<sl;p++){float dot=0;for(int d=0;d<HD;d++) dot+=q[h*HD+d]*t->kc[li][p*NC*HD+h*HD+d];sc[p]=dot/sqrtf((float)HD);}
                softmax(sc,sl);
                for(int d=0;d<HD;d++){float v=0;for(int p=0;p<sl;p++) v+=sc[p]*t->vcc[li][p*NC*HD+h*HD+d];co[h*HD+d]=v;}
                free(sc);
            }
            free(q);free(k);free(vc);
        }
        /* rrpram */
        if(NR>0){ro=calloc(NR*HD,sizeof(float));
            float *vr=calloc(NR*HD,sizeof(float));matmul(vr,xn,t->L[li].vr,D,NR*HD);
            memcpy(t->vrc[li]+pos*NR*HD,vr,NR*HD*sizeof(float));
            for(int h=0;h<NR;h++){
                float *sc=calloc(sl,sizeof(float));
                for(int p=0;p<sl;p++){float s=0;for(int d=0;d<D;d++) s+=xn[d]*t->L[li].wr[(h*D+d)*t->CTX+p];sc[p]=s;}
                softmax(sc,sl);
                for(int d=0;d<HD;d++){float v=0;for(int p=0;p<sl;p++) v+=sc[p]*t->vrc[li][p*NR*HD+h*HD+d];ro[h*HD+d]=v;}
                free(sc);
            }
            free(vr);
        }
        /* janus */
        if(NJ>0){jo=calloc(NJ*HD,sizeof(float));
            float *wjp=calloc(NJ*HD,sizeof(float)),*vjp=calloc(NJ*HD,sizeof(float));
            matmul(wjp,xn,t->L[li].wj,D,NJ*HD);matmul(vjp,xn,t->L[li].vj,D,NJ*HD);
            float norm=0;for(int d=0;d<NJ*HD;d++) norm+=wjp[d]*wjp[d];
            norm=1.0f/sqrtf(norm+1e-8f);
            for(int d=0;d<NJ*HD;d++) jo[d]=vjp[d]*(wjp[d]*norm);
            free(wjp);free(vjp);
        }
        /* gating + output */
        float *comb=calloc(D,sizeof(float));
        if(nm>1&&t->L[li].gw){
            float *gl=calloc(nm,sizeof(float));matmul(gl,xn,t->L[li].gw,D,nm);
            float gates[3];for(int g=0;g<nm;g++) gates[g]=1.0f/(1.0f+expf(-(gl[g]+t->L[li].gb[g])));
            free(gl); int off=0,gi=0;
            if(NC>0){for(int d=0;d<NC*HD;d++) comb[off+d]=gates[gi]*co[d];off+=NC*HD;gi++;}
            if(NR>0){for(int d=0;d<NR*HD;d++) comb[off+d]=gates[gi]*ro[d];off+=NR*HD;gi++;}
            if(NJ>0){for(int d=0;d<NJ*HD;d++) comb[off+d]=gates[gi]*jo[d];off+=NJ*HD;gi++;}
        }else{int off=0;
            if(NC>0&&co){memcpy(comb+off,co,NC*HD*sizeof(float));off+=NC*HD;}
            if(NR>0&&ro){memcpy(comb+off,ro,NR*HD*sizeof(float));off+=NR*HD;}
            if(NJ>0&&jo){memcpy(comb+off,jo,NJ*HD*sizeof(float));off+=NJ*HD;}
        }
        if(co)free(co);if(ro)free(ro);if(jo)free(jo);
        float *proj=calloc(D,sizeof(float));matmul(proj,comb,t->L[li].wo,D,D);
        for(int d=0;d<D;d++) x[d]=xr[d]+proj[d];
        free(proj);free(comb);
        /* mlp */
        memcpy(xr,x,D*sizeof(float));rmsnorm(xn,x,D);
        float *up=calloc(4*D,sizeof(float));matmul(up,xn,t->L[li].up,D,4*D);
        for(int d=0;d<4*D;d++) if(up[d]<0) up[d]=0;
        float *dn=calloc(D,sizeof(float));matmul(dn,up,t->L[li].dn,4*D,D);
        for(int d=0;d<D;d++) x[d]=xr[d]+dn[d];
        free(up);free(dn);
    }
    rmsnorm(xn,x,D);
    for(int v=0;v<t->V;v++){float dot=0;for(int d=0;d<D;d++) dot+=xn[d]*t->tok[v*D+d];t->logits[v]=dot;}
    /* transformer gate: magnitude-based. untrained~0.1 -> gate=0, trained~2+ -> gate=1 */
    float mag=0;for(int v=0;v<t->V;v++) mag+=fabsf(t->logits[v]);mag/=t->V;
    float tg=clampf((mag-0.5f)/1.5f,0.0f,1.0f);
    for(int v=0;v<t->V;v++) t->logits[v]*=tg;
    t->clen=sl; free(x);free(xn);free(xr);
}

/* ── coherence score ── */
static float coherence_score(const MetaW *mw, const int *ids, int n, int V){
    /* score a token sequence by average bigram probability + Hebbian density */
    if(n<2) return 0;
    float bi_sum=0,hb_sum=0;
    for(int i=0;i<n-1;i++){
        for(int j=0;j<mw->n_bi;j++){
            if(mw->bigrams[j].a==ids[i]&&mw->bigrams[j].b==ids[i+1]){bi_sum+=mw->bigrams[j].prob;break;}
        }
    }
    /* Hebbian: average association strength between adjacent pairs */
    for(int i=0;i<n-1&&i<20;i++){
        int a=ids[i]<ids[i+1]?ids[i]:ids[i+1],b=ids[i]<ids[i+1]?ids[i+1]:ids[i];
        for(int k=0;k<mw->n_hebb;k++){
            if(mw->hebbs[k].a==a&&mw->hebbs[k].b==b){hb_sum+=mw->hebbs[k].str;break;}
        }
    }
    /* progressive length bonus: strongly prefer 15+ tokens */
    float len_bonus=(n>15)?1.5f:(n>10)?0.8f:(n>6)?0.2f:-0.5f;
    return bi_sum/(n-1)+0.3f*hb_sum/(n-1)+len_bonus;
}

/* ── boundary check ── */
static int is_boundary(const BPE *bpe, int id){
    if(id<0||id>=bpe->vocab_size)return 0;
    int len=bpe->vocab_len[id];
    for(int i=0;i<len;i++){
        uint8_t c=bpe->vocab_bytes[id][i];
        if(c=='.'||c=='!'||c=='?'){
            /* only sentence boundary if last char or followed by space/newline */
            if(i==len-1) return 1;
            uint8_t nc=bpe->vocab_bytes[id][i+1];
            if(nc==' '||nc=='\n'||nc=='\r') return 1;
        }
    }
    return 0;
}

/* check if token starts with space (word boundary) */
static int starts_with_space(const BPE *bpe, int id){
    if(id<0||id>=bpe->vocab_size||bpe->vocab_len[id]==0)return 0;
    return bpe->vocab_bytes[id][0]==' ';
}

/* ── generate sentence ── */
static int gen_sent(TF *t, const BPE *bpe, MetaW *mw,
                    const int *prompt, int plen, float temp,
                    int *out, int maxo, Parliament *parl, float *global_destiny,
                    Chambers *ch_ptr){
    tf_reset(t); int V=t->V,D=t->D;
    float *destiny=calloc(D,sizeof(float));
    /* inherit global destiny direction (thematic coherence across chain) */
    if(global_destiny) for(int d=0;d<D;d++) destiny[d]=0.3f*global_destiny[d];
    float *prev_logits=calloc(V,sizeof(float));
    int prev_chosen=-1;
    int ctx[MAX_SEQ],cl=0,gl=0;
    for(int i=0;i<plen&&i<t->CTX-1;i++){tf_forward(t,prompt[i],i);ctx[cl++]=prompt[i];out[gl++]=prompt[i];}
    for(int step=0;step<120&&gl<maxo;step++){
        int pos=cl-1; if(pos>=t->CTX-1) break;
        tf_forward(t,ctx[cl-1],pos);
        float *raw=calloc(V,sizeof(float));memcpy(raw,t->logits,V*sizeof(float));
        /* DOE Parliament injection: δ in θ = ε + γ + αδ */
        if(parl){
            float *xn=calloc(D,sizeof(float));
            rmsnorm(xn,t->tok+ctx[cl-1]*D,D);
            parl_inject(parl,raw,xn,V);
            /* NOTORCH: proper prophecy debt — top-3 unfulfilled vs chosen fulfilled */
            if(step>0&&prev_chosen>=0){
                float *debt=calloc(D,sizeof(float));
                /* find top-3 from prev logits (what was "destined") */
                int top3[3]={0,0,0}; float tv[3]={-1e30f,-1e30f,-1e30f};
                for(int i=0;i<V;i++){
                    if(prev_logits[i]>tv[2]){tv[2]=prev_logits[i];top3[2]=i;
                        for(int k=1;k>=0;k--) if(tv[k+1]>tv[k]){float tmp=tv[k];tv[k]=tv[k+1];tv[k+1]=tmp;int ti=top3[k];top3[k]=top3[k+1];top3[k+1]=ti;}
                    }
                }
                /* unfulfilled prophecy for top-3 not chosen; fulfilled for chosen */
                for(int k=0;k<3;k++) if(top3[k]!=prev_chosen&&top3[k]<V)
                    for(int d=0;d<D&&top3[k]<t->V;d++) debt[d]+=0.1f*t->tok[top3[k]*D+d];
                if(prev_chosen<t->V)
                    for(int d=0;d<D;d++) debt[d]-=0.1f*t->tok[prev_chosen*D+d];
                parl_notorch(parl,xn,debt,D);
                free(debt);
                if(step%20==0) parl_lifecycle(parl);
            }
            free(xn);
        }
        memcpy(prev_logits,raw,V*sizeof(float));
        int last=ctx[cl-1];
        if(last<V) for(int d=0;d<D;d++) destiny[d]=0.9f*destiny[d]+0.1f*t->tok[last*D+d];
        float dn=0;for(int d=0;d<D;d++) dn+=destiny[d]*destiny[d];dn=sqrtf(dn+1e-10f);
        float *heb=calloc(V,sizeof(float));
        float *pro=calloc(V,sizeof(float));
        int hs=cl>8?cl-8:0; meta_hebb(mw,ctx+hs,cl-hs,heb,V);
        meta_prophecy(mw,ctx,cl,pro,V);
        /* trauma gravity: high trauma dampens all logits */
        if(ch_ptr&&ch_ptr->trauma>0.1f)
            for(int i=0;i<V;i++) raw[i]/=(1.0f+ch_ptr->trauma);
        /* detect if transformer is active via gate magnitude */
        float tmag=0;for(int v=0;v<V;v++) tmag+=fabsf(raw[v]);tmag/=(V>0?V:1);
        int has_tf=tmag>0.1f;
        /* Dario field: B + α·H + β·P + γ·D + T — stronger without weights */
        float c_heb=has_tf?0.4f:0.8f, c_pro=has_tf?0.2f:0.5f;
        float c_ds=has_tf?0.3f:0.1f, c_bg=has_tf?5.0f:15.0f, c_tg=has_tf?3.0f:10.0f;
        for(int i=0;i<V;i++){
            float bg=meta_bi(mw,ctx[cl-1],i);
            float tg=cl>=2?meta_tri(mw,ctx[cl-2],ctx[cl-1],i):1e-10f;
            float ds=0;
            if(dn>1e-8f){float en=0;for(int d=0;d<D;d++) en+=t->tok[i*D+d]*t->tok[i*D+d];
                en=sqrtf(en+1e-10f);if(en>1e-8f){float dot=0;for(int d=0;d<D;d++) dot+=destiny[d]*t->tok[i*D+d];ds=dot/(dn*en);}}
            raw[i]+=c_heb*heb[i]+c_pro*pro[i]+c_ds*ds+c_bg*bg+c_tg*tg;
            if(mw->unigram[i]<1e-6f) raw[i]-=2.0f;
            else if(mw->unigram[i]>0.01f) raw[i]-=0.3f*(mw->unigram[i]-0.01f)*100.0f;
        }
        free(heb); free(pro);
        /* repetition penalty: stronger for recent, milder for older */
        for(int ri=cl-1;ri>=0&&ri>=cl-20;ri--){
            if(ctx[ri]<V){
                float age_factor=(float)(cl-ri); /* 1=just seen, 20=old */
                float pen=0.3f+0.035f*age_factor; /* 0.335 for recent, 0.65 for old (weaker) */
                raw[ctx[ri]]*=pen;
            }
        }
        /* bigram blocking: penalize repeating the same bigram */
        if(cl>=2){for(int ri=0;ri<cl-1;ri++){
            if(ctx[ri]==ctx[cl-2]&&ctx[ri+1]<V) raw[ctx[ri+1]]*=0.2f;
        }}
        /* hybrid decode: without weights → more greedy; with weights → nucleus after greedy start */
        int ch;
        if(!has_tf){/* no transformer: greedy with slight noise */
            if(step<6){ch=0;float mx=raw[0];for(int i=1;i<V;i++) if(raw[i]>mx){mx=raw[i];ch=i;}
            }else{ch=sample_nucleus(raw,V,0.5f,0.7f);}
        }else if(step<4){
            ch=0;float mx=raw[0];for(int i=1;i<V;i++) if(raw[i]>mx){mx=raw[i];ch=i;}
        }else{ch=sample_nucleus(raw,V,temp,0.85f);}
        free(raw);
        prev_chosen=ch;
        out[gl++]=ch; ctx[cl++]=ch;
        /* word capture: online MetaWeight update (NOTORCH) */
        if(cl>=2){
            int prev=ctx[cl-2],cur=ctx[cl-1];
            /* update bigram: strengthen this transition */
            for(int i=0;i<mw->n_bi;i++){
                if(mw->bigrams[i].a==prev&&mw->bigrams[i].b==cur){mw->bigrams[i].prob+=0.005f;goto bi_done;}
            }
            if(mw->n_bi<MAX_BIGRAM){mw->bigrams[mw->n_bi].a=prev;mw->bigrams[mw->n_bi].b=cur;mw->bigrams[mw->n_bi].prob=0.01f;mw->n_bi++;}
            bi_done:;
            /* update Hebbian: co-occurrence with recent window */
            int hw=cl>6?cl-6:0;
            for(int ri=hw;ri<cl-1;ri++){
                int a=ctx[ri]<cur?ctx[ri]:cur,b=ctx[ri]<cur?cur:ctx[ri];
                float decay=1.0f/(1.0f+abs((cl-1)-ri));
                for(int k=0;k<mw->n_hebb;k++){
                    if(mw->hebbs[k].a==a&&mw->hebbs[k].b==b){mw->hebbs[k].str+=decay*0.005f;goto hb_done;}
                }
                if(mw->n_hebb<MAX_HEBBIAN){mw->hebbs[mw->n_hebb].a=a;mw->hebbs[mw->n_hebb].b=b;mw->hebbs[mw->n_hebb].str=decay*0.01f;mw->n_hebb++;}
                hb_done:;
            }
        }
        if(is_boundary(bpe,ch)&&step>8) break; /* allow longer sentences */
    }
    /* export destiny back to global (0.7 old + 0.3 new) */
    if(global_destiny) for(int d=0;d<D;d++) global_destiny[d]=0.7f*global_destiny[d]+0.3f*destiny[d];
    free(destiny);free(prev_logits); return gl;
}

/* ── SPA — Sentence Phonon Attention ── */
/* Bidirectional sentence-level attention between chain steps.
   Tokens are atoms. Sentences are phonons. Ландау's invention.
   After all 12 steps, cross-attend and identify weak sentences for reseed. */

#define SPA_DIM  32
#define SPA_NH   4
#define SPA_HD   (SPA_DIM/SPA_NH)

typedef struct{
    float W_embed[MAX_VOCAB][SPA_DIM]; /* random init, not trained */
    float r_bias[CHAIN_STEPS+1];
    float alpha;
}SPACtx;

static void spa_init(SPACtx *s, int V){
    s->alpha=0.85f;
    for(int i=0;i<V&&i<MAX_VOCAB;i++)
        for(int d=0;d<SPA_DIM;d++) s->W_embed[i][d]=0.02f*((float)rand()/RAND_MAX-0.5f);
    for(int i=0;i<=CHAIN_STEPS;i++) s->r_bias[i]=0.1f/(1.0f+i);
}

static void spa_embed_sentence(const SPACtx *s, const int *ids, int n, float *out){
    /* exponential weighted mean → project to SPA_DIM */
    memset(out,0,SPA_DIM*sizeof(float));
    if(n==0) return;
    float total_w=0;
    for(int i=0;i<n;i++){
        float w=powf(s->alpha,(float)(n-1-i));
        if(ids[i]>=0&&ids[i]<MAX_VOCAB)
            for(int d=0;d<SPA_DIM;d++) out[d]+=w*s->W_embed[ids[i]][d];
        total_w+=w;
    }
    if(total_w>0) for(int d=0;d<SPA_DIM;d++) out[d]/=total_w;
    /* normalize */
    float norm=0;for(int d=0;d<SPA_DIM;d++) norm+=out[d]*out[d];
    norm=1.0f/sqrtf(norm+1e-8f);
    for(int d=0;d<SPA_DIM;d++) out[d]*=norm;
}

static void spa_cross_attend(const SPACtx *s, float embs[][SPA_DIM], int S, float scores[]){
    /* bidirectional attention. Returns per-sentence "connectedness" score */
    for(int i=0;i<S;i++){
        float total_attn=0;
        for(int j=0;j<S;j++){
            if(i==j) continue;
            float dot=0;for(int d=0;d<SPA_DIM;d++) dot+=embs[i][d]*embs[j][d];
            dot/=sqrtf((float)SPA_DIM);
            int dist=abs(i-j);if(dist>CHAIN_STEPS) dist=CHAIN_STEPS;
            dot+=s->r_bias[dist];
            total_attn+=expf(dot);
        }
        scores[i]=total_attn; /* higher = more connected */
    }
}

/* ── chain ── */
static void gen_chain(TF *t, const BPE *bpe, MetaW *mw, Chambers *ch,
                      const int *cids, int clen, int has_weights, Parliament *parl){
    /* calendar dissonance */
    struct tm e={0};e.tm_year=2024-1900;e.tm_mon=9;e.tm_mday=3;e.tm_hour=12;
    time_t epoch=mktime(&e); float days=epoch>0?(float)difftime(time(NULL),epoch)/86400.0f:0;
    float y=days/365.25f,drift=y*11.25f; int full=(int)(y/19);float corr=full*7*30.0f;
    float partial=fmodf(y,19);int yic=(int)partial+1;
    int met[]={3,6,8,11,14,17,19};for(int i=0;i<7;i++) if(met[i]<=yic) corr+=30;
    drift-=corr; float cd=clampf(fabsf(fmodf(drift,33))/33,0,1);

    int nb=(int)(CHAIN_STEPS*(0.3f+0.4f*ch->debt+0.1f*cd));
    if(nb<1)nb=1;if(nb>=CHAIN_STEPS)nb=CHAIN_STEPS-1;

    printf("\n  diss=%.3f debt=%.3f %s\n  chambers:",cd,ch->debt,has_weights?"[TRAINED]":"[METAWEIGHTS ONLY]");
    for(int i=0;i<6;i++) if(ch->act[i]>0.05f) printf(" %s:%.0f%%",CH_N[i],ch->act[i]*100);
    if(parl) {float av=0;for(int i=0;i<parl->n;i++) av+=parl->ex[i].vitality;av/=(parl->n>0?parl->n:1);
        printf("\n  parliament: %d experts, avg_vitality=%.2f",parl->n,av);}
    printf("\n\n");

    float *gdest=calloc(t->D,sizeof(float)); /* persistent destiny across chain */
    SPACtx spa; spa_init(&spa,t->V);
    int chain_ids[CHAIN_STEPS][256]; int chain_lens[CHAIN_STEPS];
    for(int si=0;si<CHAIN_STEPS;si++){
        int dir=si<nb?-1:(si==nb?0:1);
        /* pick prompt: destiny-guided for forward steps, random for backward */
        int start=-1;
        if(dir>=0&&si>0){
            /* destiny-guided: find sentence boundary whose tokens have high destiny dot */
            float best_score=-1e30f;int best_pos=-1,tries2=0;
            while(tries2<50){
                int r=rand()%(clen>5?clen-5:1);
                if(is_boundary(bpe,cids[r])&&r+3<clen&&starts_with_space(bpe,cids[r+1])){
                    /* score: dot product of first prompt token embedding with destiny */
                    float sc=0;int tok=cids[r+1];
                    if(tok<t->V) for(int d=0;d<t->D;d++) sc+=t->tok[tok*t->D+d]*gdest[d];
                    if(sc>best_score){best_score=sc;best_pos=r+1;}
                }
                tries2++;
            }
            if(best_pos>=0) start=best_pos;
        }
        if(start<0){/* random boundary for backward steps or fallback */
            int tries=0;
            while(start<0&&tries<200){
                int r=rand()%(clen>5?clen-5:1);
                if(is_boundary(bpe,cids[r])&&r+3<clen&&starts_with_space(bpe,cids[r+1])){start=r+1;break;}
                tries++;
            }
        }
        if(start<0) start=rand()%(clen>5?clen-5:1); /* final fallback */
        int plen=start+5<=clen?5:3;
        int prompt[5]={cids[start],cids[start+1],cids[start+2],
                       plen>3?cids[start+3]:0,plen>4?cids[start+4]:0};
        /* Schumann resonance: 7.83Hz fundamental + harmonics modulate temperature */
        float t_sec=(float)si/(float)CHAIN_STEPS;
        float schumann=0.4f*sinf(2*M_PI*7.83f*t_sec)+0.2f*sinf(2*M_PI*14.3f*t_sec)
                       +0.1f*sinf(2*M_PI*20.8f*t_sec)+0.05f*sinf(2*M_PI*27.3f*t_sec);
        float base_temp=has_weights?0.6f:0.75f;
        float temp=clampf(base_temp+0.08f*schumann,0.4f,0.85f);
        /* best-of-3: generate 3 candidates, pick highest coherence */
        int best_out[256],best_ol=0; float best_sc=-1e30f;
        float gdest_save[256]; if(t->D<=256) memcpy(gdest_save,gdest,t->D*sizeof(float));
        for(int cand=0;cand<3;cand++){
            if(cand>0&&t->D<=256) memcpy(gdest,gdest_save,t->D*sizeof(float)); /* restore destiny */
            int out[256],ol=gen_sent(t,bpe,mw,prompt,plen,temp,out,256,parl,gdest,ch);
            float sc=coherence_score(mw,out,ol,t->V);
            if(sc>best_sc){best_sc=sc;best_ol=ol;memcpy(best_out,out,ol*sizeof(int));}
            if(best_sc>1.0f&&best_ol>12) break; /* early exit if first candidate is strong */
        }
        char mk=dir<0?'<':(dir==0?'*':'>');
        printf("  [%2d] %c ",si+1,mk);
        /* quality gate: skip if too short or low coherence */
        if(best_ol<5||(best_sc<0.01f&&best_ol<8)){
            printf("[...]\n");
        }else{
            char buf[128];int printed=0;
            for(int i=0;i<best_ol&&printed<200;i++){int len=bpe_decode_token(bpe,best_out[i],buf,sizeof(buf));if(len>0){printf("%s",buf);printed+=len;}}
            printf("\n");
        }
        /* save for SPA */
        chain_lens[si]=best_ol; memcpy(chain_ids[si],best_out,best_ol*sizeof(int));
        ch_xfire(ch,3); ch->debt=0.9f*ch->debt+0.05f;
    }
    /* SPA: cross-attend sentences, find weak ones, reseed */
    float spa_embs[CHAIN_STEPS][SPA_DIM]; float spa_scores[CHAIN_STEPS];
    for(int i=0;i<CHAIN_STEPS;i++) spa_embed_sentence(&spa,chain_ids[i],chain_lens[i],spa_embs[i]);
    spa_cross_attend(&spa,spa_embs,CHAIN_STEPS,spa_scores);
    /* find weakest sentence */
    float min_sc=spa_scores[0];int weak_idx=0;
    for(int i=1;i<CHAIN_STEPS;i++) if(spa_scores[i]<min_sc){min_sc=spa_scores[i];weak_idx=i;}
    float avg_sc=0;for(int i=0;i<CHAIN_STEPS;i++) avg_sc+=spa_scores[i];avg_sc/=CHAIN_STEPS;
    if(min_sc<avg_sc*0.5f){
        /* reseed the weakest sentence */
        printf("  [SPA] reseeding step %d (score=%.2f, avg=%.2f)\n",weak_idx+1,min_sc,avg_sc);
        int r=rand()%(clen>5?clen-5:1);
        int prompt[5]={cids[r],cids[r+1],cids[r+2],cids[r+3],cids[r+4]};
        int out[256],ol=gen_sent(t,bpe,mw,prompt,5,has_weights?0.55f:0.7f,out,256,parl,gdest,ch);
        printf("  [%2d] + ",weak_idx+1);
        char buf[128];int printed=0;
        for(int i=0;i<ol&&printed<200;i++){int len=bpe_decode_token(bpe,out[i],buf,sizeof(buf));if(len>0){printf("%s",buf);printed+=len;}}
        printf("\n");
    }
    /* Hebbian decay: old memories fade after each chain */
    for(int i=0;i<mw->n_hebb;i++) mw->hebbs[i].str*=0.998f;
    free(gdest);
}

/* ── main ── */
int main(int argc, char **argv){
    printf("PostGPT-Q — Resonant Reasoning Engine (C)\ntheta = epsilon + gamma + alpha*delta\nresonance is unbreakable.\n\n");
    if(argc<3){printf("Usage: %s [weights.bin] corpus.merges corpus.txt\n",argv[0]);return 1;}
    srand((unsigned)time(NULL));

    int has_weights=0; const char *wpath=NULL,*mpath,*cpath;
    if(argc>=4){wpath=argv[1];mpath=argv[2];cpath=argv[3];has_weights=1;}
    else{mpath=argv[1];cpath=argv[2];}

    printf("[1] BPE...\n");
    BPE bpe; if(!bpe_load(&bpe,mpath)) return 1;
    printf("  %d merges, vocab=%d\n",bpe.n_merges,bpe.vocab_size);

    printf("[2] Corpus...\n");
    FILE *cf=fopen(cpath,"rb"); if(!cf){fprintf(stderr,"ERROR: %s\n",cpath);return 1;}
    fseek(cf,0,SEEK_END);long csz=ftell(cf);fseek(cf,0,SEEK_SET);
    uint8_t *craw=malloc(csz);fread(craw,1,csz,cf);fclose(cf);
    int *cids=malloc(csz*sizeof(int));int clen=bpe_encode(&bpe,craw,(int)csz,cids,(int)csz);
    free(craw); printf("  %ld bytes -> %d tokens\n",csz,clen);

    printf("[3] MetaWeights...\n");
    MetaW *mw=calloc(1,sizeof(MetaW)); meta_build(mw,cids,clen,bpe.vocab_size);

    TF t={0};
    if(has_weights){
        printf("[4] Transformer...\n");
        if(!tf_load(&t,wpath)) return 1;
    }else{
        printf("[4] No weights — MetaWeights only mode\n");
        /* create minimal transformer with zero weights — gate will silence it */
        t.V=bpe.vocab_size;t.D=48;t.NH=4;t.NL=1;t.CTX=64;t.NC=2;t.NR=2;t.NJ=0;t.HD=12;
        t.tok=calloc(t.V*t.D,sizeof(float));t.pos=calloc(t.CTX*t.D,sizeof(float));
        t.L=calloc(1,sizeof(t.L[0]));
        t.L[0].wq=calloc(t.NC*t.HD*t.D,sizeof(float));t.L[0].wk=calloc(t.NC*t.HD*t.D,sizeof(float));
        t.L[0].vc=calloc(t.NC*t.HD*t.D,sizeof(float));t.L[0].wr=calloc(t.NR*t.D*t.CTX,sizeof(float));
        t.L[0].vr=calloc(t.NR*t.HD*t.D,sizeof(float));t.L[0].wo=calloc(t.D*t.D,sizeof(float));
        t.L[0].up=calloc(4*t.D*t.D,sizeof(float));t.L[0].dn=calloc(t.D*4*t.D,sizeof(float));
        t.kc=calloc(1,sizeof(float*));t.vcc=calloc(1,sizeof(float*));t.vrc=calloc(1,sizeof(float*));
        t.kc[0]=calloc(t.CTX*t.NC*t.HD,sizeof(float));t.vcc[0]=calloc(t.CTX*t.NC*t.HD,sizeof(float));
        t.vrc[0]=calloc(t.CTX*t.NR*t.HD,sizeof(float));
        t.clen=0;t.logits=calloc(t.V,sizeof(float));
    }

    /* try loading saved memory */
    {FILE *mf=fopen("q.memory","rb");
    if(mf){
        uint32_t magic;fread(&magic,4,1,mf);
        if(magic==0x514D454D){
            int nb,nt,nh;fread(&nb,4,1,mf);fread(&nt,4,1,mf);fread(&nh,4,1,mf);
            /* overlay: merge saved with corpus-derived, keeping higher probs */
            for(int i=0;i<nb&&i<MAX_BIGRAM;i++){
                int a,b;float p;fread(&a,4,1,mf);fread(&b,4,1,mf);fread(&p,4,1,mf);
                int found=0;
                for(int j=0;j<mw->n_bi;j++) if(mw->bigrams[j].a==a&&mw->bigrams[j].b==b){
                    if(p>mw->bigrams[j].prob) mw->bigrams[j].prob=p;found=1;break;}
                if(!found&&mw->n_bi<MAX_BIGRAM){mw->bigrams[mw->n_bi].a=a;mw->bigrams[mw->n_bi].b=b;mw->bigrams[mw->n_bi].prob=p;mw->n_bi++;}
            }
            /* skip trigrams and hebbian for now — just seek past them */
            for(int i=0;i<nt;i++){int tmp[3];float f;fread(tmp,4,3,mf);fread(&f,4,1,mf);}
            for(int i=0;i<nh;i++){int tmp[2];float f;fread(tmp,4,2,mf);fread(&f,4,1,mf);}
            printf("  [memory loaded: %d bi from q.memory]\n",nb);
        }
        fclose(mf);
    }}

    Chambers ch; ch_init(&ch);

    printf("[5] DOE Parliament...\n");
    Parliament parl; parl_init(&parl,t.D,4);
    printf("  %d experts, rank=%d, d_model=%d, alpha=%.2f\n",parl.n,DOE_RANK,t.D,parl.alpha);

    printf("\n========== 12 BIDIRECTIONAL STEPS ==========\n");
    gen_chain(&t,&bpe,mw,&ch,cids,clen,has_weights,&parl);

    printf("\ntype -> 12 sentences. 'quit' to exit.\n\n");
    char input[1024];
    while(1){
        printf("  q> ");if(!fgets(input,sizeof(input),stdin))break;
        input[strcspn(input,"\n")]=0;
        if(!input[0]||!strcmp(input,"quit")||!strcmp(input,"exit"))break;
        /* inject user input into MetaWeights (word capture from dialogue) */
        {int uids[512];int ulen=bpe_encode(&bpe,(const uint8_t*)input,(int)strlen(input),uids,512);
        if(ulen>1){
            /* bigram capture */
            for(int i=0;i<ulen-1;i++){
                int a=uids[i],b=uids[i+1];int found=0;
                for(int j=0;j<mw->n_bi;j++) if(mw->bigrams[j].a==a&&mw->bigrams[j].b==b){mw->bigrams[j].prob+=0.02f;found=1;break;}
                if(!found&&mw->n_bi<MAX_BIGRAM){mw->bigrams[mw->n_bi].a=a;mw->bigrams[mw->n_bi].b=b;mw->bigrams[mw->n_bi].prob=0.05f;mw->n_bi++;}
            }
            /* trigram capture */
            for(int i=0;i<ulen-2;i++){
                int a=uids[i],b=uids[i+1],c=uids[i+2];int found=0;
                for(int j=0;j<mw->n_tri;j++) if(mw->trigrams[j].a==a&&mw->trigrams[j].b==b&&mw->trigrams[j].c==c){mw->trigrams[j].prob+=0.02f;found=1;break;}
                if(!found&&mw->n_tri<MAX_TRIGRAM){mw->trigrams[mw->n_tri].a=a;mw->trigrams[mw->n_tri].b=b;mw->trigrams[mw->n_tri].c=c;mw->trigrams[mw->n_tri].prob=0.05f;mw->n_tri++;}
            }
            /* Hebbian capture */
            for(int i=0;i<ulen;i++){for(int j=(i-6>0?i-6:0);j<ulen&&j<=i+6;j++){
                if(i==j)continue;
                int ha=uids[i]<uids[j]?uids[i]:uids[j],hb=uids[i]<uids[j]?uids[j]:uids[i];
                float decay=1.0f/(1.0f+abs(i-j));int found=0;
                for(int k=0;k<mw->n_hebb;k++) if(mw->hebbs[k].a==ha&&mw->hebbs[k].b==hb){mw->hebbs[k].str+=decay*0.01f;found=1;break;}
                if(!found&&mw->n_hebb<MAX_HEBBIAN){mw->hebbs[mw->n_hebb].a=ha;mw->hebbs[mw->n_hebb].b=hb;mw->hebbs[mw->n_hebb].str=decay*0.02f;mw->n_hebb++;}
            }}
            printf("  [ingested %d tokens: +bi +tri +hebb]\n",ulen);
        }}
        /* modulate chambers based on input sentiment */
        if(strstr(input,"love")||strstr(input,"beauty")||strstr(input,"kind")) ch.act[CH_LOVE]+=0.15f;
        if(strstr(input,"fear")||strstr(input,"dark")||strstr(input,"death")) ch.act[CH_FEAR]+=0.15f;
        if(strstr(input,"rage")||strstr(input,"anger")||strstr(input,"fire")) ch.act[CH_RAGE]+=0.15f;
        if(strstr(input,"void")||strstr(input,"nothing")||strstr(input,"silence")) ch.act[CH_VOID]+=0.15f;
        if(strstr(input,"flow")||strstr(input,"water")||strstr(input,"music")) ch.act[CH_FLOW]+=0.15f;
        ch.act[CH_FLOW]+=0.1f;ch_xfire(&ch,8);
        gen_chain(&t,&bpe,mw,&ch,cids,clen,has_weights,&parl);
    }
    /* save evolved MetaWeights — Q remembers between sessions */
    {FILE *mf=fopen("q.memory","wb");
    if(mf){
        uint32_t magic=0x514D454D; /* QMEM */
        fwrite(&magic,4,1,mf);
        fwrite(&mw->n_bi,4,1,mf);fwrite(&mw->n_tri,4,1,mf);fwrite(&mw->n_hebb,4,1,mf);
        for(int i=0;i<mw->n_bi;i++){fwrite(&mw->bigrams[i].a,4,1,mf);fwrite(&mw->bigrams[i].b,4,1,mf);fwrite(&mw->bigrams[i].prob,4,1,mf);}
        for(int i=0;i<mw->n_tri;i++){fwrite(&mw->trigrams[i].a,4,1,mf);fwrite(&mw->trigrams[i].b,4,1,mf);fwrite(&mw->trigrams[i].c,4,1,mf);fwrite(&mw->trigrams[i].prob,4,1,mf);}
        for(int i=0;i<mw->n_hebb;i++){fwrite(&mw->hebbs[i].a,4,1,mf);fwrite(&mw->hebbs[i].b,4,1,mf);fwrite(&mw->hebbs[i].str,4,1,mf);}
        fclose(mf);printf("  [memory saved: %d bi, %d tri, %d hebb → q.memory]\n",mw->n_bi,mw->n_tri,mw->n_hebb);
    }}
    printf("\nresonance is unbreakable.\n");
    free(cids);free(mw);return 0;
}
