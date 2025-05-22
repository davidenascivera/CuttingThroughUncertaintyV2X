/*
    This code is based on "Computational Geometry in C" (Second Edition),
    Chapter 7, modified to be callable from Python using ctypes.

    Original code written by Joseph O'Rourke.
    Last modified: December 1997
    Questions to orourke@cs.smith.edu.
    --------------------------------------------------------------------
    Original code is Copyright 1997 by Joseph O'Rourke.
    --------------------------------------------------------------------
    */
   #include <stdio.h>
   #include <math.h>
   #include <stdlib.h>
   #define EXIT_SUCCESS 0
   #define EXIT_FAILURE 1
   #define X       0
   #define Y       1
   typedef enum { FALSE, TRUE } bool;
   typedef enum { Pin, Qin, Unknown } tInFlag;

   #define DIM     2               /* Dimension of points */
   typedef int     tPointi[DIM];   /* type integer point */
   typedef double  tPointd[DIM];   /* type double point */
   #define PMAX    1000            /* Max # of pts in polygon */

   typedef tPointi tPolygoni[PMAX];/* type integer polygon */

   /*---------------------------------------------------------------------
   Function prototypes.
   ---------------------------------------------------------------------*/
   double  Dot( tPointi a, tPointi b );
   int     AreaSign( tPointi a, tPointi b, tPointi c );
   char    SegSegInt( tPointi a, tPointi b, tPointi c, tPointi d, tPointd p, tPointd q );
   char    ParallelInt( tPointi a, tPointi b, tPointi c, tPointi d, tPointd p, tPointd q );
   bool    Between( tPointi a, tPointi b, tPointi c );
   void    Assigndi( tPointd p, tPointi a );
   void    SubVec( tPointi a, tPointi b, tPointi c );
   bool    LeftOn( tPointi a, tPointi b, tPointi c );
   bool    Left( tPointi a, tPointi b, tPointi c );
   tInFlag InOut( tPointd p, tInFlag inflag, int aHB, int bHA, double *result, int *result_count );
   int     Advance( int a, int *aa, int n, bool inside, tPointi v, double *result, int *result_count );
   /*-------------------------------------------------------------------*/

   /*
    * Main function for Python to call: Takes two polygons and returns their intersection
    * points in the result array. Returns the number of intersection points.
    */
   int __attribute__((visibility("default"))) convex_intersect(
       int *P_flat, int n, 
       int *Q_flat, int m, 
       double *result)
   {
       tPolygoni P, Q;
       int     a, b;           /* indices on P and Q (resp.) */
       int     a1, b1;         /* a-1, b-1 (resp.) */
       tPointi A, B;           /* directed edges on P and Q (resp.) */
       int     cross;          /* sign of z-component of A x B */
       int     bHA, aHB;       /* b in H(A); a in H(b). */
       tPointi Origin = {0,0}; /* (0,0) */
       tPointd p;              /* double point of intersection */
       tPointd q;              /* second point of intersection */
       tInFlag inflag;         /* {Pin, Qin, Unknown}: which inside */
       int     aa, ba;         /* # advances on a & b indices (after 1st inter.) */
       bool    FirstPoint;     /* Is this the first point? (used to initialize).*/
       tPointd p0;             /* The first point. */
       int     code;           /* SegSegInt return code. */ 
       int     result_count = 0;   /* Number of intersection points found */

       /* Convert the flat arrays to our internal format */
       for (int i = 0; i < n; i++) {
           P[i][X] = P_flat[i*2];
           P[i][Y] = P_flat[i*2+1];
       }
       
       for (int i = 0; i < m; i++) {
           Q[i][X] = Q_flat[i*2];
           Q[i][Y] = Q_flat[i*2+1];
       }

       /* Initialize variables. */
       a = 0; b = 0; aa = 0; ba = 0;
       inflag = Unknown; FirstPoint = TRUE;

       do {
           /* Computations of key variables. */
           a1 = (a + n - 1) % n;
           b1 = (b + m - 1) % m;

           SubVec( P[a], P[a1], A );
           SubVec( Q[b], Q[b1], B );

           cross = AreaSign( Origin, A, B );
           aHB   = AreaSign( Q[b1], Q[b], P[a] );
           bHA   = AreaSign( P[a1], P[a], Q[b] );
           
           /* If A & B intersect, update inflag. */
           code = SegSegInt( P[a1], P[a], Q[b1], Q[b], p, q );
           
           if ( code == '1' || code == 'v' ) {
               if ( inflag == Unknown && FirstPoint ) {
                   aa = ba = 0;
                   FirstPoint = FALSE;
                   p0[X] = p[X]; p0[Y] = p[Y];
                   
                   /* Add first intersection point to result */
                   result[result_count*2] = p0[X];
                   result[result_count*2+1] = p0[Y];
                   result_count++;
               }
               inflag = InOut(p, inflag, aHB, bHA, result, &result_count);
           }

           /*-----Advance rules-----*/

           /* Special case: A & B overlap and oppositely oriented. */
           if ( ( code == 'e' ) && (Dot( A, B ) < 0) ) {
               /* Add the shared segment endpoints to result */
               result[result_count*2] = p[X];
               result[result_count*2+1] = p[Y];
               result_count++;
               
               result[result_count*2] = q[X];
               result[result_count*2+1] = q[Y];
               result_count++;
               
               return result_count;
           }

           /* Special case: A & B parallel and separated. */
           if ( (cross == 0) && ( aHB < 0) && ( bHA < 0 ) ) {
               return 0; /* Polygons are disjoint */
           }

           /* Special case: A & B collinear. */
           else if ( (cross == 0) && ( aHB == 0) && ( bHA == 0 ) ) {
               /* Advance but do not output point. */
               if ( inflag == Pin )
                   b = Advance( b, &ba, m, inflag == Qin, Q[b], result, &result_count );
               else
                   a = Advance( a, &aa, n, inflag == Pin, P[a], result, &result_count );
           }

           /* Generic cases. */
           else if ( cross >= 0 ) {
               if ( bHA > 0)
                   a = Advance( a, &aa, n, inflag == Pin, P[a], result, &result_count );
               else
                   b = Advance( b, &ba, m, inflag == Qin, Q[b], result, &result_count );
           }
           else /* if ( cross < 0 ) */{
               if ( aHB > 0)
                   b = Advance( b, &ba, m, inflag == Qin, Q[b], result, &result_count );
               else
                   a = Advance( a, &aa, n, inflag == Pin, P[a], result, &result_count );
           }

       /* Quit when both adv. indices have cycled, or one has cycled twice. */
       } while ( ((aa < n) || (ba < m)) && (aa < 2*n) && (ba < 2*m) );

       if ( !FirstPoint ) { /* If at least one point output, close up. */
           /* Add closing point if needed and not already added */
           if (result_count > 1 && 
               (result[0] != result[(result_count-1)*2] || 
                result[1] != result[(result_count-1)*2+1])) {
               result[result_count*2] = p0[X];
               result[result_count*2+1] = p0[Y];
               result_count++;
           }
       }
       
       return result_count;
   }

   /*---------------------------------------------------------------------
   Adds the double point of intersection to result, and toggles in/out flag.
   ---------------------------------------------------------------------*/
   tInFlag InOut( tPointd p, tInFlag inflag, int aHB, int bHA, double *result, int *result_count )
   {
       /* Add intersection point to result */
       result[(*result_count)*2] = p[X];
       result[(*result_count)*2+1] = p[Y];
       (*result_count)++;
       
       /* Update inflag. */
       if ( aHB > 0)
           return Pin;
       else if ( bHA > 0)
           return Qin;
       else    /* Keep status quo. */
           return inflag;
   }

   /*---------------------------------------------------------------------
   Advances and adds an inside vertex to result if appropriate.
   ---------------------------------------------------------------------*/
   int Advance(int a, int *aa, int n, bool inside, tPointi v, double *result, int *result_count)
   {
       if (inside) {
           /* Add inside vertex to result */
           result[(*result_count)*2] = v[X];
           result[(*result_count)*2+1] = v[Y];
           (*result_count)++;
       }
       (*aa)++;
       return (a+1) % n;
   }

   bool    Left( tPointi a, tPointi b, tPointi c )
   {
           return  AreaSign( a, b, c ) > 0;
   }

   bool    LeftOn( tPointi a, tPointi b, tPointi c )
   {
           return  AreaSign( a, b, c ) >= 0;
   }

   bool    Collinear( tPointi a, tPointi b, tPointi c )
   {
           return  AreaSign( a, b, c ) == 0;
   }
   
   /*---------------------------------------------------------------------
   a - b ==> c.
   ---------------------------------------------------------------------*/
   void    SubVec( tPointi a, tPointi b, tPointi c )
   {
       int i;

       for( i = 0; i < DIM; i++ )
           c[i] = a[i] - b[i];
   }

   int     AreaSign( tPointi a, tPointi b, tPointi c )
   {
       double area2;

       area2 = ( b[0] - a[0] ) * (double)( c[1] - a[1] ) -
               ( c[0] - a[0] ) * (double)( b[1] - a[1] );

       /* The area should be an integer. */
       if      ( area2 >  0.5 ) return  1;
       else if ( area2 < -0.5 ) return -1;
       else                     return  0;
   }

   /*---------------------------------------------------------------------
   SegSegInt: Finds the point of intersection p between two closed
   segments ab and cd.  Returns p and a char with the following meaning:
   'e': The segments collinearly overlap, sharing a point.
   'v': An endpoint (vertex) of one segment is on the other segment,
           but 'e' doesn't hold.
   '1': The segments intersect properly (i.e., they share a point and
           neither 'v' nor 'e' holds).
   '0': The segments do not intersect (i.e., they share no points).
   Note that two collinear segments that share just one point, an endpoint
   of each, returns 'e' rather than 'v' as one might expect.
   ---------------------------------------------------------------------*/
   char	SegSegInt( tPointi a, tPointi b, tPointi c, tPointi d, tPointd p, tPointd q )
   {
   double  s, t;       /* The two parameters of the parametric eqns. */
   double num, denom;  /* Numerator and denoninator of equations. */
   char code = '?';    /* Return char characterizing intersection. */

   denom = a[X] * (double)( d[Y] - c[Y] ) +
           b[X] * (double)( c[Y] - d[Y] ) +
           d[X] * (double)( b[Y] - a[Y] ) +
           c[X] * (double)( a[Y] - b[Y] );

   /* If denom is zero, then segments are parallel: handle separately. */
   if (denom == 0.0)
       return  ParallelInt(a, b, c, d, p, q);

   num =    a[X] * (double)( d[Y] - c[Y] ) +
               c[X] * (double)( a[Y] - d[Y] ) +
               d[X] * (double)( c[Y] - a[Y] );
   if ( (num == 0.0) || (num == denom) ) code = 'v';
   s = num / denom;

   num = -( a[X] * (double)( c[Y] - b[Y] ) +
               b[X] * (double)( a[Y] - c[Y] ) +
               c[X] * (double)( b[Y] - a[Y] ) );
   if ( (num == 0.0) || (num == denom) ) code = 'v';
   t = num / denom;

   if      ( (0.0 < s) && (s < 1.0) &&
               (0.0 < t) && (t < 1.0) )
       code = '1';
   else if ( (0.0 > s) || (s > 1.0) ||
               (0.0 > t) || (t > 1.0) )
       code = '0';

   p[X] = a[X] + s * ( b[X] - a[X] );
   p[Y] = a[Y] + s * ( b[Y] - a[Y] );

   return code;
   }
   
   char   ParallelInt( tPointi a, tPointi b, tPointi c, tPointi d, tPointd p, tPointd q )
   {
       if ( !Collinear( a, b, c) )
           return '0';

       if ( Between( a, b, c ) && Between( a, b, d ) ) {
           Assigndi( p, c );
           Assigndi( q, d );
           return 'e';
       }
       if ( Between( c, d, a ) && Between( c, d, b ) ) {
           Assigndi( p, a );
           Assigndi( q, b );
           return 'e';
       }
       if ( Between( a, b, c ) && Between( c, d, b ) ) {
           Assigndi( p, c );
           Assigndi( q, b );
           return 'e';
       }
       if ( Between( a, b, c ) && Between( c, d, a ) ) {
           Assigndi( p, c );
           Assigndi( q, a );
           return 'e';
       }
       if ( Between( a, b, d ) && Between( c, d, b ) ) {
           Assigndi( p, d );
           Assigndi( q, b );
           return 'e';
       }
       if ( Between( a, b, d ) && Between( c, d, a ) ) {
           Assigndi( p, d );
           Assigndi( q, a );
           return 'e';
       }
       return '0';
   }
   
   void	Assigndi( tPointd p, tPointi a )
   {
       int i;
       for ( i = 0; i < DIM; i++ )
           p[i] = a[i];
   }
   
   /*---------------------------------------------------------------------
   Returns TRUE iff point c lies on the closed segement ab.
   Assumes it is already known that abc are collinear.
   ---------------------------------------------------------------------*/
   bool    Between( tPointi a, tPointi b, tPointi c )
   {
       /* If ab not vertical, check betweenness on x; else on y. */
       if ( a[X] != b[X] )
           return ((a[X] <= c[X]) && (c[X] <= b[X])) ||
                  ((a[X] >= c[X]) && (c[X] >= b[X]));
       else
           return ((a[Y] <= c[Y]) && (c[Y] <= b[Y])) ||
                  ((a[Y] >= c[Y]) && (c[Y] >= b[Y]));
   }

   /*---------------------------------------------------------------------
   Returns the dot product of the two input vectors.
   ---------------------------------------------------------------------*/
   double  Dot( tPointi a, tPointi b )
   {
       int i;
       double sum = 0.0;

       for( i = 0; i < DIM; i++ )
           sum += a[i] * b[i];

       return  sum;
   }