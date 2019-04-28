#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>

#include <cassert>
#include <iostream>
#include <fstream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

typedef CGAL::Triangulation_vertex_base_2<K>                     Vb;
typedef CGAL::Constrained_triangulation_face_base_2<K>           Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb>              TDS;
typedef CGAL::Exact_predicates_tag                               Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, TDS, Itag> CDT;
typedef CDT::Point          Point;

CDT cdt;

// int main( )
// {
//   CDT cdt;
//   std::cout << "Inserting a grid of 5x5 constraints " << std::endl;
//   for (int i = 1; i < 6; ++i)
//     cdt.insert_constraint( Point(0,i), Point(6,i));
//   for (int j = 1; j < 6; ++j)
//     cdt.insert_constraint( Point(j,0), Point(j,6));

//   assert(cdt.is_valid());
//   int count = 0;
//   for (CDT::Finite_edges_iterator eit = cdt.finite_edges_begin();
//        eit != cdt.finite_edges_end();
//        ++eit)
//     if (cdt.is_constrained(*eit)) ++count;
//   std::cout << "The number of resulting constrained edges is  ";
//   std::cout <<  count << std::endl;
//   return 0;
// }

extern "C"
int insertConstraint(int x1,int y1,int x2,int y2){
  cdt.insert_constraint(Point(x1,y1),Point(x2,y2));
}

extern "C"
int calculateCDT(){
  assert(cdt.is_valid());
  int count = 0;
  std::ofstream ofile("data/triangulations.txt");
  for (CDT::Finite_edges_iterator eit = cdt.finite_edges_begin();
       eit != cdt.finite_edges_end();
       ++eit){
    CDT::Vertex_handle f_v1=eit->first->vertex(cdt.cw(eit->second));
    CDT::Vertex_handle f_v2=eit->first->vertex(cdt.ccw(eit->second));
    Point p1=f_v1->point();
    Point p2=f_v2->point();
    ofile<<int(p1.x())<<std::endl;
    ofile<<int(p1.y())<<std::endl;
    ofile<<int(p2.x())<<std::endl;
    ofile<<int(p2.y())<<std::endl;
  }
  ofile.close();
  //std::cout << "The number of resulting constrained edges is  ";
  //std::cout <<  count << std::endl;
}
